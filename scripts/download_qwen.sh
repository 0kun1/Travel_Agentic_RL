#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# 下载 Qwen3-4B-Instruct-2507 基座模型
# ==========================================

MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
LOCAL_DIR="./pretrained/Qwen/Qwen3-4B-Instruct-2507"

mkdir -p "${LOCAL_DIR}"

echo "Model ID: ${MODEL_ID}"
echo "Local dir: ${LOCAL_DIR}"

# 推荐 AutoDL / 国内环境优先用 modelscope
pip install -U modelscope

modelscope download \
  --model "${MODEL_ID}" \
  --local_dir "${LOCAL_DIR}"

echo "Download finished."
echo "Model saved to ${LOCAL_DIR}"