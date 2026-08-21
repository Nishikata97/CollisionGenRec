"""Inference: Trie-constrained beam search + CCE evaluation.

Primary metrics: ItemHit@K, ItemNDCG@K (collision-corrected)
Reference metrics: Hit@K, NDCG@K (standard, collision-biased)

With --save_per_user, also saves beam SID sequences for post-hoc analysis.
"""

import argparse
import json
import math
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

from utils import (
    parse_global_args, parse_dataset_args, parse_test_args,
    set_seed, ensure_dir, load_datasets, load_test_dataset,
)
from collator import TestCollator
from cce import (
    get_topk_results, get_metrics_results,
    build_sid_to_group_size, compute_cce_metrics,
)
from generation_trie import Trie, prefix_allowed_tokens_fn

# ZCR provenance lookup (sidecar next to <dataset>.index.json)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tokenizer"))
from zcr_provenance import read_zcr_mode  # noqa: E402


def test(args):
    set_seed(args.seed)
    print(vars(args))

    device = torch.device("cuda", args.gpu_id)

    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(args.base_model, model_max_length=512)

    train_data, _ = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    print(f"Added {add_num} new tokens. Train samples: {len(train_data)}")

    model = T5ForConditionalGeneration.from_pretrained(
        args.ckpt_path, low_cpu_mem_usage=True,
        device_map={"": args.gpu_id},
    )

    num_return = args.num_return_sequences or min(args.num_beams, 20)
    num_return = min(num_return, args.num_beams)

    test_data = load_test_dataset(args)
    collator = TestCollator(tokenizer)
    all_items = test_data.get_all_items()

    # Build trie for constrained decoding
    candidate_trie = Trie(
        [[0] + tokenizer.encode(candidate) for candidate in all_items]
    )
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    # Build collision group sizes for ItemHit/ItemNDCG
    sid_to_group_size = build_sid_to_group_size(test_data.indices)
    n_items = len(test_data.indices)
    from collections import Counter
    _sid_strs = ["".join(v) for v in test_data.indices.values()]
    _group_sizes = Counter(_sid_strs)
    n_unique_sids = len(_group_sizes)
    _items_in_collision = sum(s for s in _group_sizes.values() if s > 1)
    _g_max = max(_group_sizes.values()) if _group_sizes else 0
    _n_collision_groups = sum(1 for s in _group_sizes.values() if s > 1)
    collision_rate = _items_in_collision / n_items if n_items > 0 else 0.0
    print(f"Items: {n_items}, Unique SIDs: {n_unique_sids}, "
          f"Collision rate: {collision_rate:.4f} "
          f"(G_max: {_g_max}, n_collision_groups: {_n_collision_groups})")

    # Parse K values from metrics arg
    metrics = args.metrics.split(",")
    K_values = sorted({int(m.split("@")[1]) for m in metrics})

    test_loader = DataLoader(
        test_data, batch_size=args.test_batch_size, collate_fn=collator,
        shuffle=False, num_workers=8, pin_memory=True,
    )
    print(f"Test samples: {len(test_data)}")
    model.eval()

    # Accumulators: primary (item-level) and reference (SID-level)
    item_metrics_agg = {}
    sid_metrics_agg = {}
    total = 0
    per_user_data = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader)):
            inputs = batch[0].to(device)
            targets = batch[1]
            total += len(targets)

            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=10,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                num_beams=args.num_beams,
                num_return_sequences=num_return,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )
            output_ids = output["sequences"]
            scores = output["sequences_scores"]
            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            # SID-level (reference) metrics
            topk_res = get_topk_results(
                decoded, scores, targets, num_return,
                all_items=all_items if args.filter_items else None,
            )
            batch_sid_metrics = get_metrics_results(topk_res, metrics)
            for m, val in batch_sid_metrics.items():
                sid_metrics_agg[m] = sid_metrics_agg.get(m, 0) + val

            # Item-level (primary) metrics - per user
            for b_idx in range(len(targets)):
                batch_seqs = [
                    s.strip().replace(" ", "")
                    for s in decoded[b_idx * num_return: (b_idx + 1) * num_return]
                ]
                batch_sc = scores[b_idx * num_return: (b_idx + 1) * num_return]
                pairs = sorted(
                    zip(batch_seqs, batch_sc.tolist()),
                    key=lambda x: x[1], reverse=True,
                )
                beam_sids = [p[0] for p in pairs]

                item_m = compute_cce_metrics(
                    beam_sids, targets[b_idx], sid_to_group_size, K_values,
                )
                for k, v in item_m.items():
                    item_metrics_agg[k] = item_metrics_agg.get(k, 0) + v

                if args.save_per_user:
                    user_record = {
                        "beam_sids": beam_sids,
                        "target": targets[b_idx],
                    }
                    # Per-user SID-level metrics
                    row = topk_res[b_idx]
                    for m in metrics:
                        k = int(m.split("@")[1])
                        res = row[:k]
                        if m.lower().startswith("hit"):
                            user_record[m] = 1.0 if sum(res) > 0 else 0.0
                        elif m.lower().startswith("ndcg"):
                            user_record[m] = sum(
                                res[i] / math.log(i + 2, 2) for i in range(len(res))
                            )
                    # Per-user item-level metrics
                    for k, v in item_m.items():
                        user_record[k] = v
                    per_user_data.append(user_record)

            if step % 50 == 0:
                preview = {k: item_metrics_agg[k] / total for k in sorted(item_metrics_agg)}
                print(f"[primary] {preview}")

    # Normalize
    for k in item_metrics_agg:
        item_metrics_agg[k] /= total
    for k in sid_metrics_agg:
        sid_metrics_agg[k] /= total

    print("=" * 60)
    print(f"Collision rate: {collision_rate:.4f}")
    print(f"[Primary]   ItemHit/ItemNDCG: {item_metrics_agg}")
    print(f"[Reference] Hit/NDCG:         {sid_metrics_agg}")
    print("=" * 60)

    # Save results: primary first, then reference
    results = {}
    for k in sorted(item_metrics_agg):
        results[k] = item_metrics_agg[k]
    for k in sorted(sid_metrics_agg):
        results[k] = sid_metrics_agg[k]

    # Resolve ZCR provenance from sidecar next to <dataset>.index.json
    # (data_path is the data root e.g. './data'; args.index_file is relative
    # to data_path/<dataset>/, e.g. 'indexes/rkmeans_zcr_cf/Beauty.index.json'.)
    zcr_mode = read_zcr_mode(
        os.path.join(args.dataset, args.index_file),
        data_root=args.data_path,
    )

    save_data = {
        "metadata": {
            "zcr_mode": zcr_mode,
            "index_file": args.index_file,
            "dataset": args.dataset,
        },
        "results": results,
        "collision_rate": collision_rate,
        "n_collision_groups": _n_collision_groups,
        "g_max": _g_max,
        "total": total,
    }
    ensure_dir(os.path.dirname(args.results_file))
    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)

    if args.save_per_user and per_user_data:
        per_user_file = args.results_file.replace(".json", "_per_user.json")
        with open(per_user_file, "w") as f:
            json.dump(per_user_data, f)
        print(f"Per-user metrics saved to {per_user_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    args = parser.parse_args()
    test(args)
