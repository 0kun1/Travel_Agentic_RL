#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# GRPO / RL 训练脚本
#
# 作用:
# 1. 使用 SFT merged model 作为 policy model
# 2. 使用同一个模型作为 ref_model
# 3. 读取 data/final/rl.jsonl 作为 RL prompts
# 4. 调用 rollout server 生成多条轨迹
# 5. 使用 reward plugin 对轨迹打分
# 6. 使用 GRPO 更新模型
#
# ==========================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# 读取 .env，里面有 OPENAI_API_KEY / OPENAI_BASE_URL / JUDGE_MODEL_ID
if [ -f "${PROJECT_ROOT}/.env" ]; then
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
fi

# Judge 配置，reward plugin 会用到
export JUDGE_API_KEY="${JUDGE_API_KEY:-${OPENAI_API_KEY:-}}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-${OPENAI_BASE_URL:-}}"
export JUDGE_MODEL="${JUDGE_MODEL:-${JUDGE_MODEL_ID:-deepseek-chat}}"
export JUDGE_TIMEOUT_SEC="${JUDGE_TIMEOUT_SEC:-30}"

# NCCL 参数
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

MS_SWIFT_DIR="${MS_SWIFT_DIR:-${PROJECT_ROOT}/ms-swift}"

# SFT merged model，RL 从这里继续训练
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/saved/output/qwen3_4b_sft_merged}"

# GRPO 的 ref_model，一般用初始 SFT 模型
REF_MODEL_PATH="${REF_MODEL_PATH:-${MODEL_PATH}}"

# RL 数据
RL_DATASET="${RL_DATASET:-${PROJECT_ROOT}/data/final/rl.jsonl}"

# reward 需要用 gold dataset 对照
ANSWER_JUDGE_GOLD_DATASET_PATH="${ANSWER_JUDGE_GOLD_DATASET_PATH:-${PROJECT_ROOT}/data/final/rl.jsonl}"
export ANSWER_JUDGE_GOLD_DATASET_PATH

# 插件路径
REWARD_PLUGIN="${REWARD_PLUGIN:-${PROJECT_ROOT}/ms-swift/examples/train/grpo/plugin/tooluse_reward_parser_aligned.py}"
SCHEDULER_PLUGIN="${SCHEDULER_PLUGIN:-${PROJECT_ROOT}/ms-swift/examples/train/grpo/plugin/tooluse_multi_turn_scheduler.py}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/saved/output/grpo_parser_aligned_run}"

# 训练 GPU 配置
# 这里默认写 0，仅用于代码结构。完整 RL 不建议单卡直接跑。
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29511}"

# rollout server 地址
VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-127.0.0.1}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"

# GRPO 训练超参
MAX_TURNS="${MAX_TURNS:-13}"
MAX_STEPS="${MAX_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-50}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

# GRPO 每个 prompt 采样多少条 completion
# 老师默认 NUM_GENERATIONS=NPROC_PER_NODE
NUM_GENERATIONS="${NUM_GENERATIONS:-${NPROC_PER_NODE}}"

TEMPERATURE="${TEMPERATURE:-0.9}"
MAX_LENGTH="${MAX_LENGTH:-50000}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-5000}"

# reward curriculum 配置，参考老师源码
export PARSER_REWARD_TOTAL_STEPS="${PARSER_REWARD_TOTAL_STEPS:-${MAX_STEPS}}"
export PARSER_REWARD_PHASE_RATIOS="${PARSER_REWARD_PHASE_RATIOS:-[0.15, 0.2, 0.65]}"

# 六个 reward 分量权重：
# 1. answer_tag / 格式
# 2. tool_call_schema / JSON 可解析
# 3. tool_process / 工具流程
# 4. stage / 不该提前 answer
# 5. efficiency / 少重复少乱调
# 6. llm_judge / 最终答案质量
export PARSER_REWARD_W1="${PARSER_REWARD_W1:-[0.10, 0.10, 0.05, 0.05, 0.10, 0.60]}"
export PARSER_REWARD_W2="${PARSER_REWARD_W2:-[0.05, 0.05, 0.05, 0.05, 0.10, 0.70]}"
export PARSER_REWARD_W3="${PARSER_REWARD_W3:-[0.05, 0.05, 0.05, 0.05, 0.05, 0.75]}"
export PARSER_REWARD_DEBUG="${PARSER_REWARD_DEBUG:-1}"

# W&B / TensorBoard
REPORT_TO="${REPORT_TO:-wandb}"
RUN_NAME="${RUN_NAME:-qwen3_4b_travel_grpo}"

if [ "${REPORT_TO}" = "wandb" ]; then
  export WANDB_PROJECT="${WANDB_PROJECT:-travel-agentic-rl}"
  export WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
  export WANDB_MODE="${WANDB_MODE:-online}"
fi

echo "========== GRPO TRAIN CONFIG =========="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MS_SWIFT_DIR=${MS_SWIFT_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "REF_MODEL_PATH=${REF_MODEL_PATH}"
echo "RL_DATASET=${RL_DATASET}"
echo "ANSWER_JUDGE_GOLD_DATASET_PATH=${ANSWER_JUDGE_GOLD_DATASET_PATH}"
echo "REWARD_PLUGIN=${REWARD_PLUGIN}"
echo "SCHEDULER_PLUGIN=${SCHEDULER_PLUGIN}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "VLLM_SERVER=${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "NUM_GENERATIONS=${NUM_GENERATIONS}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH}"
echo "REPORT_TO=${REPORT_TO}"
echo "RUN_NAME=${RUN_NAME}"
echo "======================================="

if [ ! -d "${MODEL_PATH}" ]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  echo "Please merge SFT LoRA first."
  exit 1
fi

if [ ! -f "${RL_DATASET}" ]; then
  echo "[ERROR] RL_DATASET not found: ${RL_DATASET}"
  exit 1
fi

if [ ! -f "${REWARD_PLUGIN}" ]; then
  echo "[ERROR] REWARD_PLUGIN not found: ${REWARD_PLUGIN}"
  exit 1
fi

if [ ! -f "${SCHEDULER_PLUGIN}" ]; then
  echo "[ERROR] SCHEDULER_PLUGIN not found: ${SCHEDULER_PLUGIN}"
  exit 1
fi

echo "Using swift:"
which swift
python -c "import swift; print(swift.__file__)"

CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_PORT="${MASTER_PORT}" \
swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL_PATH}" \
  --ref_model "${REF_MODEL_PATH}" \
  --dataset "${RL_DATASET}" \
  --external_plugins "${REWARD_PLUGIN}" "${SCHEDULER_PLUGIN}" \
  --reward_funcs external_parser_aligned_curriculum_reward \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host "${VLLM_SERVER_HOST}" \
  --vllm_server_port "${VLLM_SERVER_PORT}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --vllm_enable_prefix_caching true \
  --vllm_disable_custom_all_reduce true \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --deepspeed zero3 \
  --learning_rate "${LEARNING_RATE}" \
  --max_steps "${MAX_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 2 \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --max_length "${MAX_LENGTH}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --num_generations "${NUM_GENERATIONS}" \
  --temperature "${TEMPERATURE}" \
  --top_k 50 \
  --top_p 0.9 \
  --beta 0.04 \
  --loss_type grpo \
  --multi_turn_scheduler travel_tool_loop \
  --max_turns "${MAX_TURNS}" \
  --completion_length_limit_scope per_round \
  --scale_rewards group \
  --importance_sampling_level token \
  --num_iterations 1 \
  --dataloader_num_workers 1 \
  --dataset_num_proc 1 \
  --logging_steps 1 \
  --log_completions true \
  --report_to "${REPORT_TO}" \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}"