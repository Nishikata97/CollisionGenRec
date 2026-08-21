"""ZCR provenance helper.

Writes two sidecars next to <dataset>.index.json so tools can distinguish ZCR
(optimal) from greedy reassignment (both produce zero-collision indexes):

  <dataset>.zcr_provenance.json   - minimal zcr_mode marker (optimal|greedy|none)
  <dataset>.stats.json           - zcr_stats + project-standard collision_rate

Project-standard collision_rate:
    items_in_non_singleton_groups / n_items
NOT (n_items - n_unique) / n_items (which is duplicate-rate style).
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Iterable

PROVENANCE_VERSION = "1.0"

ZCR_MODE_OPTIMAL = "optimal"
ZCR_MODE_GREEDY = "greedy"
ZCR_MODE_NONE = "none"
ZCR_MODE_UNKNOWN = "unknown"
VALID_ZCR_MODES = {ZCR_MODE_OPTIMAL, ZCR_MODE_GREEDY, ZCR_MODE_NONE}


def _project_collision_rate(codes_after) -> float:
    """items_in_non_singleton_groups / n_items (project standard)."""
    n = len(codes_after)
    if n == 0:
        return 0.0
    code_tuples = (
        tuple(c.tolist() if hasattr(c, "tolist") else c) for c in codes_after
    )
    counter = Counter(code_tuples)
    items_in_non_singleton = sum(cnt for cnt in counter.values() if cnt > 1)
    return items_in_non_singleton / n


def write_zcr_sidecars(
    index_dir: str,
    dataset: str,
    zcr_mode: str,
    zcr_stats: dict,
    codes_after,
    codebook_size: int,
) -> None:
    """Write <index_dir>/<dataset>.zcr_provenance.json and .stats.json.

    Args:
        index_dir: directory containing <dataset>.index.json
        dataset: dataset name (Beauty/Yelp/Cell/Scientific)
        zcr_mode: 'optimal' | 'greedy' | 'none'
        zcr_stats: dict from apply_zcr() (n_reassigned,
            n_collision_groups, max_prefix_group_size, total_dist_increase,
            avg_dist_increase). Missing keys default to 0.
        codes_after: (n_items, L) numpy array - final codes after ZCR
            (or pre-ZCR if zcr_mode == 'none')
        codebook_size: V (used to derive prefix_feasibility)
    """
    if zcr_mode not in VALID_ZCR_MODES:
        raise ValueError(
            f"zcr_mode must be one of {sorted(VALID_ZCR_MODES)}, got {zcr_mode!r}"
        )

    os.makedirs(index_dir, exist_ok=True)

    # Provenance sidecar - minimal authoritative zcr_mode marker
    prov = {
        "zcr_mode": zcr_mode,
        "schema_version": PROVENANCE_VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(index_dir, f"{dataset}.zcr_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)

    # Stats sidecar - only fields computable from apply_zcr() return
    # + codes_after (post_zcr_collision_rate). do NOT include
    # mean_dist_increase / max_dist_increase - those are not in zcr_stats.
    max_pgs = int(zcr_stats.get("max_prefix_group_size", 0))
    stats = {
        "zcr_mode": zcr_mode,
        "n_reassigned": int(zcr_stats.get("n_reassigned", 0)),
        "n_collision_groups": int(zcr_stats.get("n_collision_groups", 0)),
        "max_prefix_group_size": max_pgs,
        "total_dist_increase": float(zcr_stats.get("total_dist_increase", 0.0)),
        "avg_dist_increase": float(zcr_stats.get("avg_dist_increase", 0.0)),
        "post_zcr_collision_rate": _project_collision_rate(codes_after),
        "prefix_feasibility": (max_pgs <= codebook_size),
        "codebook_size": codebook_size,
    }
    with open(os.path.join(index_dir, f"{dataset}.stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def read_zcr_mode(index_file_relpath: str, data_root: str = "./data") -> str:
    """Resolve zcr_mode from sidecar next to .index.json.

    Args:
        index_file_relpath: path relative to data_root, e.g.
            'Beauty/indexes/rkmeans_zcr_cf/Beauty.index.json' (this is what
            model/test.py passes - `os.path.join(args.dataset, args.index_file)`).
            A bare 'indexes/.../X.index.json' also works if data_root is
            already pointed at data_root/<dataset>/.
        data_root: root of the data tree (default './data')

    Returns 'unknown' if sidecar missing or unreadable.
    """
    full_path = os.path.join(data_root, index_file_relpath)
    index_dir = os.path.dirname(full_path)
    fname = os.path.basename(full_path)
    # fname like 'Beauty.index.json' -> dataset 'Beauty'
    dataset = fname.split(".")[0]
    sidecar = os.path.join(index_dir, f"{dataset}.zcr_provenance.json")
    if not os.path.exists(sidecar):
        return ZCR_MODE_UNKNOWN
    try:
        with open(sidecar) as f:
            return json.load(f).get("zcr_mode", ZCR_MODE_UNKNOWN)
    except (OSError, json.JSONDecodeError):
        return ZCR_MODE_UNKNOWN
