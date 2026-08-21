#!/bin/bash
# Tokenize: build a Semantic ID (SID) index for one dataset + variant.
#
#   bash scripts/tokenize.sh <DATASET> <VARIANT>
#
#   DATASET ∈ {Beauty, Scientific, Cell, Yelp}
#   VARIANT ∈ {rkmeans_native, rkmeans_zcr, rkmeans_zcr_cf,
#              rqvae_native,  rqvae_zcr,
#              letter_native, letter_zcr,
#              quasid_native, quasid_zcr}
#
# Single-GPU; honors CUDA_VISIBLE_DEVICES. Activate your environment first
# (see README "Requirements").
#
# NOTE: tokenizing from scratch needs the item embeddings
# (data/<DATASET>/<DATASET>.emb-qwen3.npy), which are NOT shipped; generate
# them with data_process/text_embedding.py. The pre-built SID indexes under
# data/<DATASET>/indexes/<VARIANT>/ are the canonical reproduction artifacts;
# you do not need to re-tokenize to reproduce the reported numbers. For the
# RQ-VAE / LETTER / QuaSID families the regenerated SIDs are not guaranteed
# bit-identical to the shipped indexes (quantizer training stochasticity).
set -e

DATASET=${1:?Usage: bash scripts/tokenize.sh <DATASET> <VARIANT>}
VARIANT=${2:?Usage: bash scripts/tokenize.sh <DATASET> <VARIANT>}

case "${VARIANT}" in
    rkmeans_native)  ARGS="--quantizer rkmeans --alpha 0.0" ;;
    rkmeans_zcr)     ARGS="--quantizer rkmeans --alpha 0.0 --zcr" ;;
    rkmeans_zcr_cf)  ARGS="--quantizer rkmeans --alpha 0.5 --zcr" ;;
    rqvae_native)    ARGS="--quantizer rqvae  --alpha 0.3 --rqvae_epochs 20000" ;;
    rqvae_zcr)       ARGS="--quantizer rqvae  --alpha 0.3 --rqvae_epochs 20000 --zcr" ;;
    letter_native)   ARGS="--quantizer letter --alpha 0.3 --rqvae_epochs 20000" ;;
    letter_zcr)      ARGS="--quantizer letter --alpha 0.3 --rqvae_epochs 20000 --zcr" ;;
    quasid_native)   ARGS="--quantizer quasid --alpha 0.3 --rqvae_epochs 20000" ;;
    quasid_zcr)      ARGS="--quantizer quasid --alpha 0.3 --rqvae_epochs 20000 --zcr" ;;
    mql4grec)
        echo "mql4grec SIDs are produced by the MQL4GRec tokenizer (external)."
        echo "The shipped index data/${DATASET}/indexes/mql4grec/ is canonical;"
        echo "this repo does not regenerate it. Run train.sh/test.sh directly."
        exit 1 ;;
    *)
        echo "Unknown variant: ${VARIANT}"; exit 1 ;;
esac

echo "Tokenize: ${DATASET} / ${VARIANT}  ->  ${ARGS} (--zcr_mode optimal by default)"
python tokenizer/build_index.py \
    --dataset "${DATASET}" \
    --output_dir "./data/${DATASET}" \
    --variant_name "${VARIANT}" \
    ${ARGS}
