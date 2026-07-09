#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# SFT LoRA 合并脚本
#
# 作用：
# 把 SFT 训练得到的 LoRA adapter
# 合并回 Qwen3-4B-Instruct-2507 基座模型，
# 得到一个完整的 SFT 模型。
# ==========================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/pretrained/Qwen/Qwen3-4B-Instruct-2507}"

# 你训练结束后要把这里改成真实 checkpoint
# 也可以运行时传入：
# ADAPTER_PATH=xxx ./scripts/merge_lora.sh
ADAPTER_PATH=/root/autodl-tmp/travel_agentic_rl_rebuild/saved/output/qwen3_4b_sft_lora/v0-20260709-234044/checkpoint-2310 \
OUTPUT_DIR=/root/autodl-tmp/travel_agentic_rl_rebuild/saved/output/qwen3_4b_sft_merged_2310 \

echo "========== MERGE LORA =========="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "ADAPTER_PATH=${ADAPTER_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "================================"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi

if [ ! -d "${ADAPTER_PATH}" ]; then
  echo "[ERROR] ADAPTER_PATH not found: ${ADAPTER_PATH}"
  echo "Please check checkpoints:"
  ls -lh "${PROJECT_ROOT}/saved/output/qwen3_4b_sft_lora" || true
  exit 1
fi

swift export \
  --model "${MODEL_PATH}" \
  --adapters "${ADAPTER_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --merge_lora true \
  --safe_serialization true

echo "[DONE] merged model saved to: ${OUTPUT_DIR}"