#!/usr/bin/env bash
# ============================================================
# DAPO Training: Qwen3-8B with FP8 training (forward + backward)
# Training: Megatron FP8 hybrid (blockwise)
# Rollout:  vLLM FP8 (KV cache quantization)
# ============================================================
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 2
pkill -9 ray || true
pkill -9 python || true


TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
EXP_NAME="tis-longtext_dapo-qwen3-8b-fp8-${TIMESTAMP}"
OUTPUT_DIR="/lanzichang1/new_verl/checkpoints/$EXP_NAME"
mkdir -p $OUTPUT_DIR

source /lanzichang1/miniconda3/bin/activate
conda activate verl_env
cd /lanzichang1/new_verl/verl

# ---- Environment ----
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.9
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_USE_DEEP_GEMM=1
export VLLM_USE_DEEP_GEMM_E8M0=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1

# ---- Sequence length ----
max_prompt_length=$((1024*2))
max_response_length=$((1024*16))
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))

# ---- DAPO algorithm parameters ----
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"

# ---- Filter groups (dynamic sampling) ----
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

# ---- Overlong buffer ----
enable_overlong_buffer=True
overlong_buffer_len=512
overlong_penalty_factor=1.0

# ---- Rollout Correction ----
rollout_is=token
rollout_is_threshold=2.0
rollout_rs=null
rollout_rs_threshold=null
rollout_rs_threshold_lower=null
rollout_token_veto_threshold=null

use_dynamic_bsz=True

PYTHONUNBUFFERED=1 python3 -m recipe.dapo.main_dapo \
    --config-name='dapo_megatron_trainer.yaml' \
    data.train_files=/lanzichang1/data/long_context/dapo-math-17k.parquet \
    data.val_files=/lanzichang1/data/long_context/aime24.parquet \
    data.train_batch_size=128 \
    data.gen_batch_size=256 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    algorithm.rollout_correction.rollout_rs_threshold_lower=${rollout_rs_threshold_lower} \
    algorithm.rollout_correction.rollout_token_veto_threshold=${rollout_token_veto_threshold} \
    actor_rollout_ref.model.path=/lanzichang1/models/Qwen3-8B \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(( 40 * 1024 )) \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.dtype=bfloat16 \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.quantization.weight_dtype=fp8 \
    actor_rollout_ref.rollout.quantization.kv_cache_dtype=fp8 \
    actor_rollout_ref.rollout.quantization.calculate_kv_scales=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$(( 40 * 1024 )) \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    reward_model.overlong_buffer.log=False \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=2 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=500 \
    trainer.log_val_generations=1 \
    trainer.logger=['swanlab'] \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    +trainer.validation_data_dir="${OUTPUT_DIR}/validation" \
    trainer.project_name='dapo-longtext-8b' \
    trainer.experiment_name=$EXP_NAME \
    2>&1 | tee $OUTPUT_DIR/verl_result.log