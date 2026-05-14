#!/usr/bin/env bash
# ============================================================
# OOL Speed Benchmark: Full FP8 (training + linear + KV cache)
# Training: Megatron FP8 hybrid (blockwise)
# Rollout:  vLLM FP8 linear weights + FP8 KV cache
# ============================================================

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
EXP_NAME="ool-fp8-all-8b-${TIMESTAMP}"
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

max_prompt_length=$((1024*2))
max_response_length=$((1024*4))
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))

# Rollout Correction parameters (same as linear_kvcache_fp8_rollout.sh)
rollout_is=token
rollout_is_threshold=2.0
rollout_rs=null
rollout_rs_threshold=null
rollout_rs_threshold_lower=null
rollout_token_veto_threshold=null

use_dynamic_bsz=True

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    data.train_files=/lanzichang1/data/math/train.parquet \
    data.val_files=/lanzichang1/data/math/test.parquet \
    data.train_batch_size=2048 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    algorithm.rollout_correction.rollout_rs_threshold_lower=${rollout_rs_threshold_lower} \
    algorithm.rollout_correction.rollout_token_veto_threshold=${rollout_token_veto_threshold} \
    actor_rollout_ref.model.path=/lanzichang1/models/Qwen3-8B-Base \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(( 40 * 1024 )) \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.dtype=bfloat16 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8=hybrid \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe=blockwise \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.fp8_recipe=blockwise \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.quantization.weight_dtype=fp8 \
    actor_rollout_ref.rollout.quantization.kv_cache_dtype=fp8 \
    actor_rollout_ref.rollout.quantization.calculate_kv_scales=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$(( 40 * 1024 )) \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=2 \
    trainer.logger=['swanlab'] \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    +trainer.validation_data_dir="${OUTPUT_DIR}/validation" \
    trainer.project_name='speed-benchmark-ool-8b-4-22' \
    trainer.experiment_name=$EXP_NAME \
    trainer.total_epochs=30 2>&1 | tee $OUTPUT_DIR/verl_result.log
