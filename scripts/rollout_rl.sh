#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# GRPO Rollout Server 启动脚本
#
# 作用:
# 1. 启动 swift rollout / vLLM server
# 2. 加载 SFT merged model
# 3. 使用 travel_tool_loop scheduler
# 4. 在 rollout 阶段真实执行工具调用循环
#
# ==========================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# 本地 ms-swift 目录
MS_SWIFT_DIR="${MS_SWIFT_DIR:-${PROJECT_ROOT}/ms-swift}"

# SFT 合并后的模型路径
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/saved/output/qwen3_4b_sft_merged}"

# 多轮工具调用 scheduler 插件
SCHEDULER_PLUGIN="${SCHEDULER_PLUGIN:-${PROJECT_ROOT}/ms-swift/examples/train/grpo/plugin/tooluse_multi_turn_scheduler.py}"

# rollout server GPU 配置
ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-0}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"

# vLLM 配置
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
MAX_TURNS="${MAX_TURNS:-13}"
MAX_LENGTH="${MAX_LENGTH:-50000}"

# NCCL 参数
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

echo "========== ROLLOUT SERVER CONFIG =========="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MS_SWIFT_DIR=${MS_SWIFT_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "SCHEDULER_PLUGIN=${SCHEDULER_PLUGIN}"
echo "ROLLOUT_CUDA_VISIBLE_DEVICES=${ROLLOUT_CUDA_VISIBLE_DEVICES}"
echo "VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE}"
echo "VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}"
echo "MAX_TURNS=${MAX_TURNS}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "==========================================="

if [ ! -d "${MODEL_PATH}" ]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  echo "Please merge SFT LoRA first."
  exit 1
fi

if [ ! -f "${SCHEDULER_PLUGIN}" ]; then
  echo "[ERROR] SCHEDULER_PLUGIN not found: ${SCHEDULER_PLUGIN}"
  exit 1
fi

echo "Using swift:"
which swift
python -c "import swift; print(swift.__file__)"

CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES}" \
swift rollout \
  --model "${MODEL_PATH}" \
  --external_plugins "${SCHEDULER_PLUGIN}" \
  --multi_turn_scheduler travel_tool_loop \
  --vllm_use_async_engine true \
  --vllm_max_model_len "${MAX_LENGTH}" \
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max_turns "${MAX_TURNS}"