# Gradient Experiments (`grad_exp`)

本目录用于**量化评估低精度(FP8)rollout 训练相对于 BF16 基线的梯度误差**,以及对比不同 rollout 修正策略(Vanilla / TIS)对梯度质量的影响。

核心问题:在 RL(GRPO/DAPO)训练中,用 **FP8 量化的 vLLM rollout** 代替 BF16 会给策略梯度带来多大扰动?**TIS(Truncated Importance Sampling,截断重要性采样)** 修正能否把这种扰动压下去?

---

## 1. 实验原理

对每一种配置,脚本会:

1. 从数据集采样若干 batch 的 prompt;
2. 用 vLLM 做一次 rollout(BF16 或 FP8),计算 reward / old_log_prob /(可选)rollout 修正 / advantage;
3. 做 **forward + backward** 得到策略梯度,但**从不更新模型**(no optimizer step);
4. 把 TP/PP/DP 合并后的梯度(HF 参数名)按 batch 累积,用 **Welford 在线算法**统计出每个参数的**逐元素梯度均值 `grad_mean` 和方差 `grad_var`**;
5. 分析阶段把某个 FP8 配置的 `(mean, var)` 与 BF16 基线对比,逐层计算误差指标,输出 Markdown 报告。

把梯度看作随机变量:**均值**反映期望更新方向(偏差 bias),**方差**反映噪声(variance)。三种配置共享同一 BF16 基线,因此可逐层公平对比。

### 三种 rollout 配置的区别

| 配置 | rollout 精度 | rollout 修正 (`rollout_is` / `threshold`) | 含义 |
|------|------|------|------|
| **BF16**(基线) | BF16 | `null` / `null` | 高精度参考,无量化 |
| **FP8-Vanilla** | FP8 | `token` / `1e9` | FP8 量化,IS 阈值设极大 ≈ **不截断**(朴素 FP8) |
| **FP8-TIS** | FP8 | `token` / `2.0` | FP8 量化 + **TIS 截断修正**(比值裁剪到 2.0) |

> Vanilla 与 TIS 的唯一区别就是 `rollout_is_threshold`:`1e9`(等于不修正) vs `2.0`(截断修正)。

---

## 2. 目录结构

```
grad_exp/
├── README.md                       # 本文件
│
├── grad_collect.py                 # 【采集主程序】rollout+fwd/bwd,Welford 统计 mean/var
├── welford.py                      # Welford 在线均值/方差累积器(支持断点续跑)
├── compare_gradient_stats.py       # 【分析】基于 mean/var,算 L2 / Cosine / Noise,出报告
│
├── grad_collect_bf16.sh            # 采集:BF16 基线   (20K response, DAPO 数据)
├── grad_collect_fp8_tis.sh         # 采集:FP8 + TIS
├── grad_collect_fp8_vanilla.sh     # 采集:FP8 朴素
│
├── math/                           # 另一组实验:math 数据集 + 6K response + 20 batch
│   ├── grad_collect_bf16.sh
│   ├── grad_collect_fp8.sh
│   ├── grad_collect_fp8_tis.sh
│   ├── grad_collect_fp8_vanilla.sh
│   └── results/                    # 该组的分析报告 (report_fp8_{base,tis,vanilla}.md)
│
├── ori_grad/                       # 原始梯度采集(只存第一个 batch 的原始 grad,不做 Welford)
│   ├── grad_collect_raw.py
│   └── grad_collect_*_raw.sh
├── analyze_raw_gradients.py        # 【分析】基于原始梯度,逐层看 std/分位数/离群点
│
├── report_tis.md                   # 顶层这组的报告(2-Wasserstein 指标)
├── report_vanilla.md
└── report_fp8_vanilla_vs_tis.md
```

> ⚠️ **注意指标不一致**:`compare_gradient_stats.py` 用 **L2 距离**(见 `math/results/`);顶层的 `report_tis.md`/`report_vanilla.md` 用的是 **2-Wasserstein 距离**(由其他版本分析脚本生成)。两套报告的第一列**不可跨套比较**(量级差 10+ 个数量级)。详见第 6 节。

---

## 3. 环境准备

采集脚本依赖 verl + Megatron + vLLM 全套环境(脚本内已写死):

```bash
source /lanzichang1/miniconda3/bin/activate
conda activate verl_env
# CUDA 12.9, 4×GPU, Qwen3-8B-Base
```

关键路径(按需修改脚本顶部变量):
- 模型:`/lanzichang1/models/Qwen3-8B-Base`
- 数据:`/lanzichang1/data/long_context/dapo-math-17k.parquet`(顶层) 或 `/lanzichang1/math/train.parquet`(math/)
- 梯度输出:`GRAD_DIR`(脚本内,如 `/lanzichang1/new_verl/grads/fp8_tis`)

---

## 4. 运行步骤

### Step 1 — 采集三组梯度统计(各跑一次,产出 mean/var)

必须**先跑 BF16 基线**,再跑两个 FP8 配置:

```bash
cd /lanzichang1/new_verl/scripts/grad_exp

bash grad_collect_bf16.sh          # -> $GRAD_DIR/bf16/{grad_mean.pt, grad_var.pt}
bash grad_collect_fp8_tis.sh       # -> $GRAD_DIR/fp8_tis/{grad_mean.pt, grad_var.pt}
bash grad_collect_fp8_vanilla.sh   # -> $GRAD_DIR/fp8_vanilla/{grad_mean.pt, grad_var.pt}
```

每个脚本会:
- rollout `NUM_GRAD_BATCHES` 个 batch(顶层=200,`math/`=20),每 batch `BATCH_SIZE=32` 个 prompt、`rollout.n=16`;
- 每 batch 的合并梯度先存成 `batch_XXXX.pt`,**折叠进 Welford 后立即删除**(省磁盘);
- 结束时在 `GRAD_DIR` 写出最终的 `grad_mean.pt`、`grad_var.pt`、`grad_stats_meta.json`。

> **断点续跑**:Welford 状态(`welford_mean.pt` / `welford_M2.pt` / `welford_meta.json`)实时落盘,中断后重跑会自动从 `n>0` 处续上。最终 finalize 后这些中间文件会被清掉。

### Step 2 — 生成对比报告

用 `compare_gradient_stats.py` 把某个 FP8 配置与 BF16 基线对比:

```bash
python compare_gradient_stats.py \
    --baseline /lanzichang1/new_verl/grads/bf16 \
    --target   /lanzichang1/new_verl/grads/fp8_tis \
    --method   FP8-TIS \
    --output   report_fp8_tis.md

python compare_gradient_stats.py \
    --baseline /lanzichang1/new_verl/grads/bf16 \
    --target   /lanzichang1/new_verl/grads/fp8_vanilla \
    --method   FP8-Vanilla \
    --output   report_fp8_vanilla.md
```

参数:
| 参数 | 说明 |
|------|------|
| `--baseline` | BF16 基线目录(含 `grad_mean.pt` / `grad_var.pt`) |
| `--target`   | 待评估的 FP8 目录 |
| `--method`   | 报告标题里显示的方法名 |
| `--output`   | 输出 Markdown 路径 |

---

## 5. 报告指标含义

`compare_gradient_stats.py` 逐层输出三个指标:

| 指标 | 公式 | 衡量 | 越好 |
|------|------|------|------|
| **L2 Dist (Means)** | `‖μ_base − μ_target‖₂` | 梯度均值的绝对偏移量(大小+方向) | 越小 |
| **Cosine Sim (Means)** | `μ_base·μ_target / (‖μ_base‖‖μ_target‖)`(整向量) | 梯度均值的**方向一致性** [-1,1] | 越接近 1 |
| **Relative Noise Exp (%)** | `(σ²_target − σ²_base)/σ²_base × 100%` | 相对**方差(噪声)膨胀率** | 越接近 0 / 负 |

`analyze_raw_gradients.py`(配合 `ori_grad/`)则给出更细的逐元素统计:std、分位数(p1~p99)、>3σ 离群点比例等,用于定位"为什么某些 norm 层方差异常"。

### 解读注意事项
- **LayerNorm 类权重**(`*_norm.weight`、`input_layernorm` 等)参数少、梯度尺度小,相对噪声膨胀的分母极小,容易出现 `+10000%` 级别的夸张值——**评估优劣应以 attn/mlp 大矩阵层为主**。
- **Cosine 普遍偏低(0~0.2)是共性**:一阶矩(均值)信噪比天然低,单看余弦不宜过度解读,**三指标合看**更稳妥。
- 采样批数越多(200 > 20),mean/var 估计越稳;批数太少时噪声膨胀指标极不稳定。

---

## 6. 两组实验为何结果差异巨大(重要)

`math/results/` 与顶层 `report_*.md` 看起来差别极大,**主要不是数据集造成的**,而是三个因素叠加:

| 维度 | 顶层 `report_*.md` | `math/results/` |
|------|------|------|
| 数据集 | `dapo-math-17k` | `math/train.parquet` |
| response 长度 | 20K | 6K |
| 采样批数 | 200 | 20 |
| 第一列指标 | **2-Wasserstein** | **L2** |
| 量级 | 1e-15 ~ 1e-13 | 1e-4 ~ 1e-2 |

1. **指标不同(L2 vs 2-Wasserstein)** → 制造了 10+ 个数量级的"假性巨大差异",跨套不可比;
2. **采样批数(20 vs 200)** → 批数少时方差估计极不稳定,是 Noise 指标符号/量级反转的主因;
3. **DAPO 数据集 + 20K 长序列** → 真实但相对次要地改变了梯度分布形状。

> **若要干净隔离"数据集"的影响**:必须固定同一分析指标、同一采样批数,只切换数据集。否则差异是三因素混杂的。

---

## 7. 典型结论(供参考)

在统一 BF16 基线下,三种 FP8 方案的梯度保真度排序通常为:

```
FP8-TIS  >  FP8-Vanilla / FP8-Base   （取决于采集配置与指标)
```

- **TIS** 的截断修正能显著减小梯度均值偏移(L2 最小),并抑制方差膨胀;
- **Vanilla**(无修正)在部分配置下方差膨胀严重,尤其 self-attn 投影层;
- 具体数值受数据集、序列长度、采样批数影响很大,**请以同一套配置内���的对比为准**。

---

## 8. 输出文件清单

每次采集(`GRAD_DIR`)产出:
- `grad_mean.pt` — `dict[str, Tensor]`,逐参数梯度均值(bf16)
- `grad_var.pt` — `dict[str, Tensor]`,逐参数梯度方差(bf16,样本方差 /(n-1))
- `grad_stats_meta.json` — `{n, sample_variance}`

分析产出:对应的 `report_*.md`。
