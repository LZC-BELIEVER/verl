#!/usr/bin/env bash
# ============================================================
# Worst-5 Benchmark: FP8 Baseline (no IS correction)
# 运行5次，每次3小时训练，记录最后验证得分
# ============================================================

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
BASE_EXP_NAME="worst5-fp8-baseline-8b-${TIMESTAMP}"
BASE_OUTPUT_DIR="/lanzichang1/new_verl/checkpoints-worst5-8b/$BASE_EXP_NAME"
mkdir -p $BASE_OUTPUT_DIR

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

max_prompt_length=$((1024*2))
max_response_length=$((1024*4))
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))
use_dynamic_bsz=True

# 训练时间限制：3小时 = 10800秒
TRAIN_TIME_LIMIT=10800

# 小batch size用于快速完成step
TRAIN_BATCH_SIZE=32
PPO_MINI_BATCH_SIZE=32

# 创建结果汇总文件
SUMMARY_FILE="${BASE_OUTPUT_DIR}/summary.txt"
echo "Worst-5 Benchmark: FP8 Baseline" > $SUMMARY_FILE
echo "Started at: $(date)" >> $SUMMARY_FILE
echo "========================================" >> $SUMMARY_FILE

# 运行5次实验
for RUN_ID in 1 2 3 4 5; do
    echo "=========================================="
    echo "Starting Run ${RUN_ID}/5 at $(date)"
    echo "=========================================="

    RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/run_${RUN_ID}"
    mkdir -p $RUN_OUTPUT_DIR

    # 强制清理GPU和Ray进程
    echo "Cleaning up GPU and Ray processes..."
    pkill -9 python 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    sleep 10

    # 记录开始时间
    START_TIME=$(date +%s)

    # 启动训练进程（使用timeout直接包裹训练命令）
    timeout --signal=SIGTERM --kill-after=30 ${TRAIN_TIME_LIMIT} \
    python3 -m verl.trainer.main_ppo \
        --config-path=config \
        --config-name='ppo_megatron_trainer.yaml' \
        data.train_files=/lanzichang1/data/math/train.parquet \
        data.val_files=/lanzichang1/data/math/test.parquet \
        data.train_batch_size=${TRAIN_BATCH_SIZE} \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        algorithm.adv_estimator=grpo \
        algorithm.rollout_correction.rollout_is=null \
        algorithm.rollout_correction.rollout_is_threshold=null \
        algorithm.rollout_correction.rollout_rs=null \
        algorithm.rollout_correction.rollout_rs_threshold=null \
        algorithm.rollout_correction.rollout_rs_threshold_lower=null \
        algorithm.rollout_correction.rollout_token_veto_threshold=null \
        actor_rollout_ref.model.path=/lanzichang1/models/Qwen3-8B-Base \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
        actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(( 40 * 1024 )) \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
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
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
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
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$(( 20 * 1024 )) \
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=2 \
        trainer.logger=['swanlab'] \
        trainer.default_local_dir="${RUN_OUTPUT_DIR}" \
        +trainer.validation_data_dir="${RUN_OUTPUT_DIR}/validation" \
        trainer.project_name='worst5-benchmark-8b' \
        trainer.experiment_name="${BASE_EXP_NAME}_run${RUN_ID}" \
        trainer.total_epochs=999999 2>&1 | tee $RUN_OUTPUT_DIR/train.log

    TRAIN_EXIT_CODE=$?
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo "Training finished after ${ELAPSED}s with exit code: $TRAIN_EXIT_CODE"

    # 如果是timeout终止（退出码124），强制清理Ray进程
    if [ $TRAIN_EXIT_CODE -eq 124 ]; then
        echo "Training was terminated by timeout, cleaning up Ray processes..."
        ray stop --force 2>/dev/null || true
        pkill -9 python 2>/dev/null || true
    fi
    echo "Run ${RUN_ID}: Training time ${ELAPSED}s, exit code ${TRAIN_EXIT_CODE}" >> $SUMMARY_FILE

    # 等待Ray和GPU资源完全释放
    echo "Waiting for resources to be released..."
    ray stop --force 2>/dev/null || true
    sleep 30

    # 从validation目录中提取最后一次验证准确率
    echo "Extracting last validation accuracy..."
    LAST_VAL_STEP=$(ls -1 ${RUN_OUTPUT_DIR}/validation/*.jsonl 2>/dev/null | \
        sed 's/.*\///;s/\.jsonl//' | sort -n | tail -1)

    if [ -n "$LAST_VAL_STEP" ]; then
        LAST_VAL_ACC=$(python3 -c "
import json
total = 0
correct = 0
with open('${RUN_OUTPUT_DIR}/validation/${LAST_VAL_STEP}.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        total += 1
        correct += data.get('acc', 0)
print(f'{correct/total:.4f}' if total > 0 else '0.0000')
" 2>/dev/null)
        echo "Last validation (step ${LAST_VAL_STEP}): $LAST_VAL_ACC"
    else
        echo "ERROR: No validation files found for run ${RUN_ID}"
        LAST_VAL_ACC="0.0"
    fi

    if [ -z "$LAST_VAL_ACC" ]; then
        LAST_VAL_ACC="0.0"
    fi
    echo "$LAST_VAL_ACC" > ${RUN_OUTPUT_DIR}/final_score.txt
    echo "Run ${RUN_ID}: Final validation accuracy = ${LAST_VAL_ACC}" >> $SUMMARY_FILE

    echo "Run ${RUN_ID} completed at $(date)"
    echo ""
done

# 汇总所有运行的得分
echo "========================================" | tee -a $SUMMARY_FILE
echo "Summary of all 5 runs:" | tee -a $SUMMARY_FILE
echo "========================================" | tee -a $SUMMARY_FILE

SCORES_ARRAY=()
for RUN_ID in 1 2 3 4 5; do
    SCORE=$(cat ${BASE_OUTPUT_DIR}/run_${RUN_ID}/final_score.txt 2>/dev/null || echo "N/A")
    echo "Run ${RUN_ID}: ${SCORE}" | tee -a $SUMMARY_FILE
    if [ "$SCORE" != "N/A" ]; then
        SCORES_ARRAY+=($SCORE)
    fi
done

# 计算统计信息
if [ ${#SCORES_ARRAY[@]} -gt 0 ]; then
    python3 << EOF | tee -a $SUMMARY_FILE
import sys
scores = [${SCORES_ARRAY[@]}]
if scores:
    import statistics
    print(f"\nStatistics:")
    print(f"  Mean:  {statistics.mean(scores):.4f}")
    print(f"  Stdev: {statistics.stdev(scores) if len(scores) > 1 else 0:.4f}")
    print(f"  Min:   {min(scores):.4f}")
    print(f"  Max:   {max(scores):.4f}")
    print(f"  Worst (Min): {min(scores):.4f}")
EOF
else
    echo "No valid scores found" | tee -a $SUMMARY_FILE
fi

echo "" | tee -a $SUMMARY_FILE
echo "Completed at: $(date)" | tee -a $SUMMARY_FILE
echo "All runs completed! Results saved to: $SUMMARY_FILE"

bash /lanzichang1/occupy.sh