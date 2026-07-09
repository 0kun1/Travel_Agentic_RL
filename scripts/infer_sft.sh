#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# SFT 模型多轮工具调用推理测试
#
# 作用：
# 用合并后的 SFT 模型跑 test_final.jsonl，
# 检查模型是否能：
# 1. 输出 <tool_call>
# 2. 工具调用 JSON 可解析
# 3. 工具能真实执行
# 4. 最l_loop_infer.py
# ==========================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/saved/output/qwen3_4b_sft_merged}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/final/test_final.jsonl}"
TOOLS_DIR="${TOOLS_DIR:-${PROJECT_ROOT}/tools}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/outputs/output_qwen3_4b_sft.jsonl}"

START_IDX="${START_IDX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"

echo "========== SFT INFER CONFIG =========="
echo "MODEL_DIR=${MODEL_DIR}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "TOOLS_DIR=${TOOLS_DIR}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "START_IDX=${START_IDX}"
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "======================================"

python run_tool_loop_infer.py \
  --model_dir "${MODEL_DIR}" \
  --dataset_path "${DATASET_PATH}" \
  --tools_dir "${TOOLS_DIR}" \
  --start_idx "${START_IDX}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_turns 13 \
  --max_new_tokens 3000 \
  --temperature 0.2 \
  --top_p 0.95 \
  --top_k 50 \
  --tool_first_enforce \
  --save_full_messages \
  --output_path "${OUTPUT_PATH}"