# Accuracy Benchmark: KV-Linear 8B (Batch Size 2K)

本目录包含8个不同rollout矫正方法的benchmark脚本，用于探究不同方法对8B模型训练精度的影响。

## 脚本列表

| 脚本名称 | 方法描述 | Rollout量化 | IS矫正方法 | 特殊配置 |
|---------|---------|------------|-----------|---------|
| `bf16_baseline.sh` | BF16全精度基线 | 无 (BF16) | 无 | on-policy参考 |
| `fp8_baseline.sh` | FP8 Rollout基线 | FP8 | 无 | 无IS矫正 |
| `vanilla_is.sh` | Vanilla IS | FP8 | token-level, 无截断 | threshold=1e9 |
| `tis.sh` | 截断重要性采样 | FP8 | token-level, TIS | threshold=2.0 |
| `qat.sh` | 量化感知训练 | FP8 | 无 | W8A8 fake-quant |
| `acr.sh` | 自适应裁剪范围 | FP8 | ACR | 动态上界, bypass_mode=False |
| `ice_pop.sh` | IcePop | FP8 | token-level | 硬信任域 [0.5, 2.0] |
| `fp8_e2e.sh` | FP8端到端 | FP8 | 无 | FP8训练+FP8推理 |

## 统一基本参数

所有脚本使用相同的基本配置（来自 speed-benchmark-ool-8b-bsz-2k）：

- **模型**: Qwen3-8B-Base
- **训练批次**: 2048
- **Mini batch**: 1024
- **Prompt长度**: 2048 tokens
- **Response长度**: 4096 tokens
- **并行配置**: TP=2 (actor/ref), TP=1 (rollout)
- **Rollout采样数**: n=8
- **GPU内存利用率**: 0.8
- **优化器卸载**: 启用
- **分布式检查点**: 启用
- **环境**: verl_env
- **项目名**: accuracy-benchmark-kv-linear-8b-4-25

## 方法配置详情

### 1. BF16 Baseline
- 完全BF16精度，无量化
- 无IS矫正（on-policy参考）
- 作为精度上限参考

### 2. FP8 Baseline
- FP8 rollout量化
- 无IS矫正
- 展示纯量化误差影响

### 3. Vanilla IS
- Token-level重要性采样
- 无截断（threshold=1e9）
- 可能存在高方差问题

### 4. TIS (Truncated IS)
- Token-level重要性采样
- 截断阈值C=2.0
- 降低方差但引入偏差

### 5. QAT (Quantization-Aware Training)
- 训练时使用FP8 fake-quantization
- W8A8模式，匹配vLLM FP8推理
- 通过训练适应量化误差

### 6. ACR (Adaptive Clipping Range)
- 动态调整PPO裁剪上界
- 上界 = (1+ε)/r_it
- 防止TIS过度裁剪导致训练崩溃
- 需要3个策略（rollout, old, current）

### 7. IcePop
- Token-level IS with硬信任域
- 权重范围外直接置零 [0.5, 2.0]
- 比TIS更激进的截断策略

### 8. FP8 E2E (End-to-End)
- 训练和推理都使用FP8
- Hybrid模式 + blockwise scaling
- 需要CUDA 12.9+和特殊环境变量
- 理论上消除训练-推理分布偏移

## 使用方法

运行单个实验：
```bash
cd /lanzichang1/new_verl/scripts/accuracy-benchmark-kv-linear-8b-bsz-2k
./bf16_baseline.sh
```

批量运行所有实验：
```bash
for script in *.sh; do
    ./"$script"
done
```

## 输出位置

- 检查点: `/lanzichang1/new_verl/checkpoints/accuracy-{method}-8b-{timestamp}/`
- 日志: `{checkpoint_dir}/verl_result.log`
- 验证数据: `{checkpoint_dir}/validation/`
- SwanLab项目: `accuracy-benchmark-kv-linear-8b-4-25`

## 注意事项

1. **FP8 E2E**: 需要设置 `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`
2. **ACR**: 需要额外的 `policy_loss.loss_mode=acr` 配置
3. **QAT**: 启用fake-quantization会增加训练时间
4. **内存**: 所有配置已针对4xGPU优化，确保足够显存

## 实验目标

通过对比这8种方法，评估：
1. 不同IS矫正方法对训练稳定性的影响
2. 量化误差对最终模型精度的影响
3. 训练-推理分布偏移的矫正效果
4. 各方法的收敛速度和最终性能
