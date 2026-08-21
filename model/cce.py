"""CCE (Collision-Corrected Evaluation) metrics for sequential recommendation.

Standard metrics:
  - Hit@K:  binary indicator of target item in top-K
  - NDCG@K: normalized discounted cumulative gain at K

Collision-corrected metrics (paper Sec. 3.1):
  - ItemHit@K:  accounts for collision group size in rank computation
  - ItemNDCG@K: accounts for collision group size in NDCG computation

ItemHit@K corrects the bias where colliding items share a single beam slot:
  For a beam SID at rank r with collision group size G_r,
  the probability that the target item is the one retrieved = 1/G_r.
"""

import math
from collections import defaultdict


def get_topk_results(predictions, scores, targets, k, all_items=None):
    """Rank predictions by score and produce per-user binary result lists.

    Args:
        predictions: flat list of decoded SID strings (B * k entries)
        scores: flat tensor/list of scores (B * k entries)
        targets: list of target SID strings (B entries)
        k: num_return_sequences per user
        all_items: if provided, demote predictions not in this set

    Returns:
        list of lists: each inner list has k binary values (1=hit, 0=miss)
    """
    results = []
    B = len(targets)
    predictions = [p.strip().replace(" ", "") for p in predictions]

    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    for b in range(B):
        batch_preds = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = sorted(zip(batch_preds, batch_scores),
                        key=lambda x: x[1], reverse=True)
        target = targets[b]
        results.append([1 if p[0] == target else 0 for p in pairs])

    return results


def get_metrics_results(topk_results, metrics):
    """Compute aggregate metrics over all users."""
    res = {}
    for m in metrics:
        k = int(m.split("@")[1])
        if m.lower().startswith("hit"):
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            res[m] = ndcg_k(topk_results, k)
    return res


def hit_k(topk_results, k):
    return sum(1.0 for row in topk_results if sum(row[:k]) > 0)


def ndcg_k(topk_results, k):
    ndcg = 0.0
    for row in topk_results:
        for i, val in enumerate(row[:k]):
            ndcg += val / math.log(i + 2, 2)
    return ndcg


# ── ItemHit@K / ItemNDCG@K (paper Sec. 3.1) ─────────────────────────────────────

def build_sid_to_group_size(index_dict):
    """Build SID string → collision group size mapping (call once, reuse per user).

    Args:
        index_dict: {item_id: ["<a_X>", ...]} mapping
    Returns:
        defaultdict(int) mapping SID string → number of items sharing it
    """
    sid_to_group_size = defaultdict(int)
    for tokens in index_dict.values():
        sid_str = "".join(tokens) if isinstance(tokens, list) else tokens
        sid_to_group_size[sid_str] += 1
    return sid_to_group_size


def compute_cce_metrics(beam_sids, target_sid, sid_to_group_size, K_values):
    """Compute collision-aware ItemHit@K and ItemNDCG@K for one user.

    Args:
        beam_sids: ranked list of SID strings from beam search
        target_sid: target item's SID string
        sid_to_group_size: precomputed from build_sid_to_group_size()
        K_values: list of K values (e.g., [5, 10])

    Returns:
        dict with ItemHit@K and ItemNDCG@K for each K
    """
    results = {}

    # Find target's beam rank
    target_rank = None
    for r, sid in enumerate(beam_sids):
        if sid == target_sid:
            target_rank = r
            break

    for K in K_values:
        if target_rank is None:
            results[f"ItemHit@{K}"] = 0.0
            results[f"ItemNDCG@{K}"] = 0.0
            continue

        G_target = sid_to_group_size.get(target_sid, 1)

        # P_r = number of items ranked before target's beam position
        P_r = sum(sid_to_group_size.get(beam_sids[j], 1)
                  for j in range(target_rank))

        # How many slots remain for target's group
        slots = max(0, K - P_r)
        effective = min(G_target, slots)

        results[f"ItemHit@{K}"] = effective / G_target

        # ItemNDCG@K
        ndcg = 0.0
        for i in range(1, effective + 1):
            ndcg += 1.0 / math.log2(P_r + i + 1)
        results[f"ItemNDCG@{K}"] = ndcg / G_target

    return results
