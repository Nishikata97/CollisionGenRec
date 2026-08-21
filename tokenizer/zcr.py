"""Zero-Collision Reassignment (ZCR) for SID tokenizers.

Items sharing the same length-(L-1) prefix form a prefix group. ZCR keeps the
first L-1 codes of every item and reassigns only the last-level code so that
all items in a prefix group receive distinct codes.

Two modes:
  - optimal (ZCR, the paper's method): min-cost bipartite matching inside each
    colliding prefix group via scipy.optimize.linear_sum_assignment, with
    lexicographic priority "minimize reassignment count, then minimize total
    distance increase, then deterministic tie-break".
  - greedy: nearest-unused-code reassignment, used as the cost baseline.

Both modes leave a prefix group unchanged (with a warning) when its size
exceeds the codebook size V.

API is model-agnostic: accepts centroids list + latent vectors, works with
both RK-Means (latents = original embeddings) and RQ-VAE (latents = MLP-
encoded representations in codebook space).
"""

from collections import defaultdict

import numpy as np
import torch


def apply_zcr(codes, centroids, latents, codebook_size, mode='optimal'):
    """Resolve SID collisions by reassigning last-level codes.

    Args:
        codes: (n_items, L) codes - numpy array or torch tensor
        centroids: list of L tensors, each (V, dim) - codebook centroids
        latents: (n_items, dim) vectors in the quantization space (numpy)
        codebook_size: V
        mode: 'optimal' (default; ZCR via min-cost bipartite matching with
              lex priority) or 'greedy' (baseline).

    Returns:
        codes: (n_items, L) collision-free numpy array
        stats: resolution statistics
    """
    if mode not in ('greedy', 'optimal'):
        raise ValueError(f"Unknown mode: {mode!r}; expected 'greedy' or 'optimal'")
    if isinstance(codes, torch.Tensor):
        codes = codes.numpy().copy()
    else:
        codes = np.array(codes, copy=True)

    n_items, L = codes.shape
    V = codebook_size

    # Compute residuals at last level (subtract centroids for levels 0..L-2)
    x = torch.tensor(latents, dtype=torch.float32)
    with torch.no_grad():
        for l in range(L - 1):
            c = centroids[l].detach()
            if c.is_cuda:
                c = c.cpu()
            x = x - c[torch.tensor(codes[:, l], dtype=torch.long)]

    # Squared distances to all last-level centroids
    centroids_last = centroids[L - 1].detach()
    if centroids_last.is_cuda:
        centroids_last = centroids_last.cpu()
    x_sq = x.pow(2).sum(dim=1, keepdim=True)
    c_sq = centroids_last.pow(2).sum(dim=1, keepdim=True).T
    sq_dists = (x_sq + c_sq - 2 * x @ centroids_last.T).numpy()

    # Group by prefix
    prefix_to_items = defaultdict(list)
    for i in range(n_items):
        prefix = tuple(codes[i, :L - 1].tolist())
        prefix_to_items[prefix].append(i)

    n_reassigned = 0
    total_dist_increase = 0.0
    max_prefix_size = 0
    n_collision_groups = 0

    for prefix, items in prefix_to_items.items():
        if len(items) <= 1:
            continue
        max_prefix_size = max(max_prefix_size, len(items))

        last_codes = [int(codes[i, L - 1]) for i in items]
        if len(set(last_codes)) == len(items):
            continue  # no collision in this group

        if len(items) > V:
            print(f"  WARNING: prefix {prefix} has {len(items)} items > V={V}")
            continue

        n_collision_groups += 1

        if mode == 'greedy':
            # Map code -> items
            code_to_items = defaultdict(list)
            for idx in items:
                code_to_items[int(codes[idx, L - 1])].append(idx)

            used_codes = set()
            losers = []

            for code, group in code_to_items.items():
                if len(group) == 1:
                    used_codes.add(code)
                else:
                    # Closest to centroid keeps the code
                    dists = [(idx, sq_dists[idx, code]) for idx in group]
                    dists.sort(key=lambda t: t[1])
                    used_codes.add(code)
                    losers.extend(dists[1:])

            # Greedy reassignment (smallest penalty first)
            losers.sort(key=lambda t: t[1])
            for item_idx, _ in losers:
                old_code = int(codes[item_idx, L - 1])
                item_dists = sq_dists[item_idx].copy()
                for uc in used_codes:
                    item_dists[uc] = np.inf
                new_code = int(np.argmin(item_dists))

                total_dist_increase += sq_dists[item_idx, new_code] - sq_dists[item_idx, old_code]
                codes[item_idx, L - 1] = new_code
                used_codes.add(new_code)
                n_reassigned += 1

        else:  # mode == 'optimal'
            # ZCR: lex bipartite matching within prefix group.
            # Cost: C(i, c) = M * 1[c != c_i^0] + sq_dist(i, c) + eps * tau(i, c)
            # where M dominates any distance span (lex priority "min reassign count first")
            # and tau is column-aware (item_rank * V + c) so ties are broken deterministically.
            from scipy.optimize import linear_sum_assignment

            items_arr = np.array(items, dtype=np.int64)
            g = items_arr.shape[0]
            initial_codes_group = codes[items_arr, L - 1].astype(np.int64)
            dist_slice = sq_dists[items_arr, :].astype(np.float64)  # (g, V)

            # Indicator matrix: 1 iff column c differs from item's initial code
            indicator = (np.arange(V, dtype=np.int64)[None, :]
                         != initial_codes_group[:, None]).astype(np.float64)
            # M strictly larger than any total distance span possible in this group
            M = float(V) * (float(dist_slice.max()) + 1e-6)

            # Column-aware lex tie-break (NOT row-constant). Use item-id rank so
            # the perturbation is invariant to the order in which `items` was built.
            sorted_perm = np.argsort(items_arr)
            item_rank = np.empty(g, dtype=np.int64)
            item_rank[sorted_perm] = np.arange(g, dtype=np.int64)
            tau = item_rank[:, None] * V + np.arange(V, dtype=np.int64)[None, :]

            cost = M * indicator + dist_slice + 1e-12 * tau.astype(np.float64)
            row_ind, col_ind = linear_sum_assignment(cost)

            for k in range(g):
                item_idx = int(items_arr[row_ind[k]])
                old_code = int(initial_codes_group[row_ind[k]])
                new_code = int(col_ind[k])
                if new_code != old_code:
                    total_dist_increase += float(
                        sq_dists[item_idx, new_code] - sq_dists[item_idx, old_code]
                    )
                    codes[item_idx, L - 1] = new_code
                    n_reassigned += 1

    stats = {
        "n_reassigned": n_reassigned,
        "n_collision_groups": n_collision_groups,
        "total_dist_increase": float(total_dist_increase),
        "avg_dist_increase": float(total_dist_increase / max(n_reassigned, 1)),
        "max_prefix_group_size": max_prefix_size,
    }
    return codes, stats
