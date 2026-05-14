TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
EXP_NAME="bf16-base-verl-${TIMESTAMP}"
OUTPUT_DIR="/lanzichang/new_verl/checkpoints/$EXP_NAME"
mkdir -p $OUTPUT_DIR

source /lanzichang/conda_lzc/etc/profile.d/conda.sh
conda activate verl_env

cd /lanzichang/new_verl/verl

export VLLM_USE_V1=1
export VLLM_USE_DEEP_GEMM=1
export VLLM_USE_DEEP_GEMM_E8M0=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export HYDRA_FULL_ERROR=1

max_prompt_length=$((1024*2))
max_response_length=$((1024*2))
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))

use_dynamic_bsz=True

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=/lanzichang/data/math/train.parquet \
 data.val_files=/lanzichang/data/math/test.parquet \
 data.train_batch_size=1 \
 data.max_prompt_length=${max_prompt_length} \
 data.max_response_length=${max_response_length} \
 algorithm.adv_estimator=grpo \
 actor_rollout_ref.model.path=/lanzichang/models/Qwen3-8B-Base \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=1 \
 actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(( 32 * 1024 )) \
 actor_rollout_ref.actor.strategy=fsdp2 \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
 actor_rollout_ref.rollout.n=4 \
 actor_rollout_ref.rollout.enforce_eager=False \
 actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$(( 32 * 1024 )) \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.save_freq=40 \
 trainer.test_freq=5 \
 trainer.logger=['swanlab'] \
 trainer.default_local_dir="${OUTPUT_DIR}" \
 trainer.validation_data_dir="${OUTPUT_DIR}/validation" \
 trainer.project_name='WTQ-Qwen3-8B-3-21' \
 trainer.experiment_name=$EXP_NAME \
 trainer.total_epochs=5 2>&1 | tee $OUTPUT_DIR/verl_result.log