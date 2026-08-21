# CollisionGenRec

PyTorch implementation of our CIKM 2026 paper:

> **Faithful Evaluation of Semantic-ID Tokenizers for Generative Recommendation**
> Qian Zhang, Lech Szymanski, Jeremiah D. Deng, and Haibo Zhang.
> *Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26).*
> [doi:10.1145/3799682.3841124](https://doi.org/10.1145/3799682.3841124)

It provides:

- **CCE** (Collision-Corrected Evaluation): item-level metrics ItemHit@K and
  ItemNDCG@K. See `model/cce.py`.
- **ZCR** (Zero-Collision Reassignment): minimum-cost last-level reassignment that
  removes SID collisions for any tokenizer. See `tokenizer/zcr.py`.

## Requirements

```
pip install -r requirements.txt
```

Main dependencies: `torch`, `transformers`, `accelerate`, `sentencepiece`,
`numpy`, `scipy` (Hungarian matching for ZCR), `scikit-learn`, `faiss-cpu`
(RK-Means), `k-means-constrained` (LETTER). `faiss-cpu` and
`k-means-constrained` are only needed to run `scripts/tokenize.sh`; training
and evaluation on the pre-built indexes do not need them. Tested versions are
listed in `requirements.txt`.

## Data

`data/<DATASET>/` (DATASET: `Beauty`, `Scientific`, `Cell`, `Yelp`) contains the
interactions `<DATASET>.inter.json`, item text `<DATASET>.item.json`, and pre-built
SID indexes `indexes/<VARIANT>/<DATASET>.index.json`. Item embeddings and trained
checkpoints are not included; the indexes alone are enough to reproduce our results.

VARIANT is one of:

```
rkmeans_native  rkmeans_zcr  rkmeans_zcr_cf
rqvae_native    rqvae_zcr
letter_native   letter_zcr
quasid_native   quasid_zcr
mql4grec
```

`*_native` is the tokenizer as-is, `*_zcr` adds ZCR, `rkmeans_zcr_cf` adds
collaborative fusion. `rqvae` is a TIGER-style RQ-VAE. `mql4grec` is collision-free
by design; its SIDs come from the external
[MQL4GRec](https://github.com/zhaijianyang/MQL4GRec) tokenizer, so we include its
indexes but not its code.

## Tokenizer

```
bash scripts/tokenize.sh <DATASET> <VARIANT>
```

Builds `indexes/<VARIANT>/<DATASET>.index.json`. This step needs the item embeddings
(not included), so use the pre-built indexes to skip it. There is no tokenizer for
`mql4grec` here; train and evaluate it on its pre-built index directly.

## Train and evaluate

```
bash scripts/train.sh <DATASET> <VARIANT>
bash scripts/test.sh  <DATASET> <VARIANT>
```

`train.sh` trains the T5 generator on the SID index; `test.sh` runs constrained
beam search and reports ItemHit@{5,10} and ItemNDCG@{5,10} (CCE), with Hit@K and
NDCG@K as reference metrics.

## Citation

```bibtex
@inproceedings{zhang2026faithful,
  title     = {Faithful Evaluation of Semantic-ID Tokenizers for Generative Recommendation},
  author    = {Zhang, Qian and Szymanski, Lech and Deng, Jeremiah D. and Zhang, Haibo},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM)},
  year      = {2026},
  doi       = {10.1145/3799682.3841124}
}
```

## Acknowledgements

The training and inference pipeline builds on [LETTER](https://github.com/HonghuiBao2000/LETTER);
the RK-Means tokenizer is adapted from [OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec),
and the MQL4GRec indexes come from the [MQL4GRec](https://github.com/zhaijianyang/MQL4GRec) tokenizer.
