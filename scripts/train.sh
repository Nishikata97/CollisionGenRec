#!/bin/bash
# Train the T5 generator on a SID index.
#
#   bash scripts/train.sh <DATASET> <VARIANT>
#
# Reads data/<DATASET>/indexes/<VARIANT>/<DATASET>.index.json (shipped) and
# writes a checkpoint to checkpoint/<DATASET>/<VARIANT>/. Single-GPU; honors
# CUDA_VISIBLE_DEVICES. Activate your environment first (see README).
set -e

DATASET=${1:?Usage: bash scripts/train.sh <DATASET> <VARIANT>}
VARIANT=${2:?Usage: bash scripts/train.sh <DATASET> <VARIANT>}

INDEX_FILE="indexes/${VARIANT}/${DATASET}.index.json"
OUTPUT_DIR="./checkpoint/${DATASET}/${VARIANT}"

echo "Train: ${DATASET} / ${VARIANT}  (index ${INDEX_FILE})"
python model/finetune.py \
    --dataset "${DATASET}" \
    --index_file "${INDEX_FILE}" \
    --data_path "./data" \
    --base_model "./configs/T5-d128" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 200 \
    --per_device_batch_size 256 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --max_his_len 20 \
    --logging_step 10
