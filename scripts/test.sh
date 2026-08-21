#!/bin/bash
# Evaluate a trained generator and print CCE metrics.
#
#   bash scripts/test.sh <DATASET> <VARIANT>
#
# Reads checkpoint/<DATASET>/<VARIANT>/ and the SID index, runs Trie-constrained
# beam search, and writes results/<DATASET>/<VARIANT>.json. The printed/saved
# metrics are the CCE metrics ItemHit@{5,10} and ItemNDCG@{5,10} (primary),
# plus SID-level Hit@K / NDCG@K (reference). Single-GPU; honors
# CUDA_VISIBLE_DEVICES. Compare the printed numbers against the paper tables.
set -e

DATASET=${1:?Usage: bash scripts/test.sh <DATASET> <VARIANT>}
VARIANT=${2:?Usage: bash scripts/test.sh <DATASET> <VARIANT>}

INDEX_FILE="indexes/${VARIANT}/${DATASET}.index.json"
CKPT_DIR="./checkpoint/${DATASET}/${VARIANT}"
RESULTS_FILE="./results/${DATASET}/${VARIANT}.json"

echo "Test: ${DATASET} / ${VARIANT}"
python model/test.py \
    --dataset "${DATASET}" \
    --index_file "${INDEX_FILE}" \
    --data_path "./data" \
    --base_model "./configs/T5-d128" \
    --ckpt_path "${CKPT_DIR}" \
    --results_file "${RESULTS_FILE}" \
    --num_beams 20 \
    --test_batch_size 64 \
    --metrics "hit@5,hit@10,ndcg@5,ndcg@10"
