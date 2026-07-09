#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# SFT 冷启动训练脚本
# 适配本地 ms-swift 源码版
#
# 注意：
# 这个版本不使用 --config yaml，
# 因为当前本地 ms-swift 没有正确读取 yaml，
# 所以所有参数都显式写在命令行里。
# ==========================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

export WANDB_PROJECT="${WANDB_PROJECT:-travel-agentic-rl}"
export WANDB_NAME="${WANDB_NAME:-qwen3_4b_travel_sft_lora}"
export WANDB_MODE="${WANDB_MODE:-online}"

MODEL_PATH="${PROJECT_ROOT}/pretrained/Qwen/Qwen3-4B-Instruct-2507"
TRAIN_DATA="${PROJECT_ROOT}/data/final/sft_train.jsonl"
VAL_DATA="${PROJECT_ROOT}/data/final/sft_val.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/saved/output/qwen3_4b_sft_lora"

echo "========== SFT CONFIG =========="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_DATA=${TRAIN_DATA}"
echo "VAL_DATA=${VAL_DATA}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_NAME=${WANDB_NAME}"
echo "================================"

echo "Using swift:"
which swift
python -c "import swift; print(swift.__file__)"

swift sft \
  --model "${MODEL_PATH}" \
  --dataset "${TRAIN_DATA}" \
  --val_dataset "${VAL_DATA}" \
  --template qwen3_nothinking \
  --tuner_type lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --target_modules all-linear \
  --num_train_epochs 2 \
  --learning_rate 8e-6 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.1 \
  --optim adamw_torch \
  --max_length 16384 \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing true \
  --dataloader_num_workers 4 \
  --save_steps 200 \
  --eval_steps 50 \
  --logging_steps 5 \
  --logging_first_step true \
  --report_to wandb \
  --run_name qwen3_4b_travel_sft_lora \
  --output_dir "${OUTPUT_DIR}"