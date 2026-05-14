#!/usr/bin/env bash
set -xeuo pipefail

# 确保脚本退出时总是执行 occupy.sh
trap 'bash /lanzichang1/occupy.sh' EXIT

# ============================================================================
# 环境配置 - 适配本地环境
# ============================================================================
source /lanzichang1/miniconda3/bin/activate
conda activate verl_env
cd /lanzichang1/new_verl/verl

# 添加时间戳避免覆盖
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)

# ============================================================================
# 项目配置
# ============================================================================
project_name='DAPO-FP8-ROLLOUT'
exp_name="DAPO-Qwen3-MOE-30B-VLLM-FP8-ROLLOUT-${TIMESTAMP}"

# ============================================================================
# 算法参数 - 保持原始配置
# ============================================================================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Rollout Correction parameters for FP8 rollout
rollout_is=token
rollout_is_threshold=2.0
rollout_rs=null
rollout_rs_threshold=null
rollout_rs_threshold_lower=null
rollout_token_veto_threshold=null

max_prompt_length=$((1024))
max_response_length=$((1024 * 20))
enable_overlong_buffer=True
overlong_buffer_len=512
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

# ============================================================================
# Batch Size 配置 - 方案 B（激进，避免OOM）
# ============================================================================
# 大幅减小batch size以适应30B MoE模型
train_prompt_bsz=4
n_resp_per_prompt=4
train_prompt_mini_bsz=4
gen_prompt_bsz=12

# ============================================================================
# 路径配置 - 适配本地路径
# ============================================================================
WORKING_DIR="/lanzichang1/new_verl/verl"
echo "WORKING_DIR: ${WORKING_DIR}"

NNODES=1  # 本地单节点
echo "NNODES: ${NNODES}"

# 模型和数据路径
MODEL_PATH="/lanzichang1/models/Qwen3-30B-A3B-Base"
CKPTS_DIR="/lanzichang1/new_verl/checkpoints/${project_name}/${exp_name}"
TRAIN_FILE="/lanzichang1/data/dapo-math-17k.parquet"
TEST_FILE="/lanzichang1/data/aime-2024.parquet"

# 创建输出目录
mkdir -p "${CKPTS_DIR}"

# ============================================================================
# 采样参数
# ============================================================================
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=1.0

# ============================================================================
# 性能和并行配置 - 适配 4 GPU，避免OOM
# ============================================================================
sp_size=1          # 禁用序列并行以节省显存
gen_tp=4           # 使用全部4个GPU做推理
train_tp=1
train_pp=1

use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=true

# ============================================================================
# 环境变量
# ============================================================================
export VERL_LOGGING_LEVEL=DEBUG
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_CONFIGURE_LOGGING=1
export VLLM_USE_V1=1
export VLLM_USE_DEEP_GEMM=1
export VLLM_USE_DEEP_GEMM_E8M0=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export HYDRA_FULL_ERROR=1

# ============================================================================
# 执行训练 - 改为直接执行（不使用 Ray job submit）
# ============================================================================
echo "=========================================="
echo "开始训练: ${exp_name}"
echo "模型: ${MODEL_PATH}"
echo "训练数据: ${TRAIN_FILE}"
echo "测试数据: ${TEST_FILE}"
echo "输出目录: ${CKPTS_DIR}"
echo "Batch配置: train_bsz=${train_prompt_bsz}, n_resp=${n_resp_per_prompt}, gen_bsz=${gen_prompt_bsz}"
echo "并行配置: sp_size=${sp_size}, gen_tp=${gen_tp}"
echo "=========================================="

PYTHONUNBUFFERED=1 python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    actor_rollout_ref.nccl_timeout=1800 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    algorithm.rollout_correction.rollout_rs_threshold_lower=${rollout_rs_threshold_lower} \
    algorithm.rollout_correction.rollout_token_veto_threshold=${rollout_token_veto_threshold} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(( 1024 * 32 )) \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.quantization.weight_dtype=fp8 \
    actor_rollout_ref.rollout.quantization.kv_cache_dtype=fp8 \
    actor_rollout_ref.rollout.quantization.calculate_kv_scales=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    reward_model.overlong_buffer.log=False \
    trainer.logger='["swanlab"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=100 \
    trainer.total_epochs=100 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=1 \
    trainer.total_training_steps=500 \
    trainer.max_actor_ckpt_to_keep=5 \
    +trainer.dump_high_diff_tokens=False \
    +trainer.dump_high_diff_dir="${CKPTS_DIR}/30B_logprob_diff_dumps" \
    actor_rollout_ref.rollout.enforce_eager=False \
    2>&1 | tee "${CKPTS_DIR}/verl_result.log"

echo "=========================================="
echo "训练完成！"
echo "日志文件: ${CKPTS_DIR}/verl_result.log"
echo "检查点目录: ${CKPTS_DIR}"
echo "=========================================="

bash /lanzichang1/occupy.sh