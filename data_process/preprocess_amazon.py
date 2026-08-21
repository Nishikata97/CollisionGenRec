"""Amazon Review data preprocessing.

Pipeline: raw Amazon gzip → 5-core filtering → inter.json + item.json

Supports Amazon 2014 (He & McAuley) and 2018 (Ni et al.) datasets.
Usage:
    python preprocess_amazon.py --dataset Beauty --version 2014 --raw_dir ../raw_data --out_dir ../data
    python preprocess_amazon.py --dataset Instruments --version 2018 --raw_dir ../raw_data --out_dir ../data
"""

import argparse
import collections
import gzip
import html
import json
import os
import re
import sys
import urllib.request
from pathlib import Path


# ── Amazon category name mappings ──────────────────────────────────────

# Short name → full category name per version
AMAZON_2014_CATEGORIES = {
    "Beauty": "Beauty",
    "Sports": "Sports_and_Outdoors",
    "Toys": "Toys_and_Games",
    "Electronics": "Electronics",
    "CDs": "CDs_and_Vinyl",
    "Clothing": "Clothing_Shoes_and_Jewelry",
    "Cell": "Cell_Phones_and_Accessories",
    "Home": "Home_and_Kitchen",
    "Movies": "Movies_and_TV",
    "Pet": "Pet_Supplies",
    "Tools": "Tools_and_Home_Improvement",
    "Automotive": "Automotive",
    "Garden": "Patio_Lawn_and_Garden",
    "Office": "Office_Products",
    "Games": "Video_Games",
    "Food": "Grocery_and_Gourmet_Food",
    "Arts": "Arts_Crafts_and_Sewing",
    "Kindle": "Kindle_Store",
    "Books": "Books",
    "Instruments": "Musical_Instruments",
}

AMAZON_2018_CATEGORIES = {
    "Beauty": "All_Beauty",
    "Fashion": "AMAZON_FASHION",
    "Appliances": "Appliances",
    "Arts": "Arts_Crafts_and_Sewing",
    "Automotive": "Automotive",
    "Books": "Books",
    "CDs": "CDs_and_Vinyl",
    "Cell": "Cell_Phones_and_Accessories",
    "Clothing": "Clothing_Shoes_and_Jewelry",
    "Electronics": "Electronics",
    "Food": "Grocery_and_Gourmet_Food",
    "Gift": "Gift_Cards",
    "Home": "Home_and_Kitchen",
    "Instruments": "Musical_Instruments",
    "Kindle": "Kindle_Store",
    "Luxury": "Luxury_Beauty",
    "Magazine": "Magazine_Subscriptions",
    "Movies": "Movies_and_TV",
    "Music": "Digital_Music",
    "Office": "Office_Products",
    "Garden": "Patio_Lawn_and_Garden",
    "Pantry": "Prime_Pantry",
    "Pet": "Pet_Supplies",
    "Scientific": "Industrial_and_Scientific",
    "Software": "Software",
    "Sports": "Sports_and_Outdoors",
    "Tools": "Tools_and_Home_Improvement",
    "Toys": "Toys_and_Games",
    "Games": "Video_Games",
}


# ── Download ───────────────────────────────────────────────────────────

def _download_file(url: str, dest: str) -> None:
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading {url}")
    print(f"  → {dest}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()  # newline after progress
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(f"Download failed: {url}\n{e}") from e


def download_amazon(dataset: str, version: str, raw_dir: str) -> dict:
    """Download raw Amazon review and metadata files.

    Args:
        dataset: Short dataset name (e.g. 'Beauty', 'Instruments').
        version: '2014' or '2018'.
        raw_dir: Directory to save raw gzip files.

    Returns:
        dict with keys 'reviews_file' and 'meta_file' (local paths).
    """
    if version == "2014":
        categories = AMAZON_2014_CATEGORIES
        base_url = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    elif version == "2018":
        categories = AMAZON_2018_CATEGORIES
        # HTTPS has TLS issues on jmcauley.ucsd.edu; use HTTP
        base_url = "http://jmcauley.ucsd.edu/data/amazon_v2"
    else:
        raise ValueError(f"Unsupported version: {version}. Use '2014' or '2018'.")

    if dataset not in categories:
        available = ", ".join(sorted(categories.keys()))
        raise ValueError(
            f"Unknown dataset '{dataset}' for Amazon {version}. "
            f"Available: {available}"
        )

    full_name = categories[dataset]
    dest_dir = os.path.join(raw_dir, f"amazon{version}")
    os.makedirs(dest_dir, exist_ok=True)

    if version == "2014":
        reviews_url = f"{base_url}/reviews_{full_name}_5.json.gz"
        meta_url = f"{base_url}/meta_{full_name}.json.gz"
        reviews_file = os.path.join(dest_dir, f"reviews_{full_name}_5.json.gz")
        meta_file = os.path.join(dest_dir, f"meta_{full_name}.json.gz")
    else:  # 2018
        reviews_url = f"{base_url}/categoryFilesSmall/{full_name}_5.json.gz"
        meta_url = f"{base_url}/metaFiles2/meta_{full_name}.json.gz"
        reviews_file = os.path.join(dest_dir, f"{full_name}_5.json.gz")
        meta_file = os.path.join(dest_dir, f"meta_{full_name}.json.gz")

    print(f"Downloading Amazon {version} — {dataset} ({full_name})")
    _download_file(reviews_url, reviews_file)
    _download_file(meta_url, meta_file)

    return {"reviews_file": reviews_file, "meta_file": meta_file}


# ── Data loading ───────────────────────────────────────────────────────

def _parse_gzip_jsonl(path: str):
    """Parse a gzip JSON-lines file (one JSON object per line)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Amazon 2014 meta files use Python repr, not JSON
                try:
                    yield eval(line)
                except Exception:
                    continue


def _load_reviews(path: str, version: str):
    """Load reviews → list of (user_raw, item_raw, timestamp, file_line_num).

    Tracks file line number so that ID assignment can follow raw file order
    (matching P5/LETTER convention).

    Field names differ between versions:
      2014: reviewerID, asin, unixReviewTime
      2018: reviewerID, asin, unixReviewTime
    """
    interactions = []
    for line_num, record in enumerate(_parse_gzip_jsonl(path)):
        user = record.get("reviewerID")
        item = record.get("asin")
        ts = record.get("unixReviewTime", 0)
        if user and item:
            interactions.append((user, item, int(ts), line_num))
    print(f"  Loaded {len(interactions)} reviews from {os.path.basename(path)}")
    return interactions


def _load_metadata(path: str):
    """Load item metadata → {asin: {title, description, brand, categories}}.

    Stores all four fields following LC-Rec convention.
    Handles format differences between Amazon 2014 and 2018:
      - 2014: 'categories' is nested list [[cat1, cat2, ...]], 'description' is str
      - 2018: 'category' is flat list [cat1, cat2, ...], 'description' is list of str
    """
    meta = {}
    for record in _parse_gzip_jsonl(path):
        asin = record.get("asin")
        if not asin:
            continue

        title = _clean_text(record.get("title", ""))

        desc = record.get("description", "")
        desc = _clean_text(desc)

        brand = record.get("brand", "")
        if not brand or not isinstance(brand, str):
            brand = ""
        else:
            brand = brand.replace("by\n", "").strip()

        # 2014 uses 'categories' (nested list), 2018 uses 'category' (flat list)
        raw_cats = record.get("categories", record.get("category", []))
        if raw_cats and isinstance(raw_cats, list):
            flat = []
            for entry in raw_cats:
                if isinstance(entry, list):
                    flat.extend(entry)
                elif isinstance(entry, str):
                    if "</span>" in entry:
                        break
                    flat.append(entry.strip())
            categories = ",".join(flat)
        else:
            categories = ""

        meta[asin] = {
            "title": title,
            "description": desc,
            "brand": brand,
            "categories": categories,
        }
    print(f"  Loaded metadata for {len(meta)} items from {os.path.basename(path)}")
    return meta


def _clean_text(raw_text) -> str:
    """Clean item metadata text - matches LETTER's utils.clean_text() exactly."""
    if isinstance(raw_text, list):
        parts = []
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r'</?\w+[^>]*>', '', raw)
            raw = re.sub(r'["\n\r]*', '', raw)
            parts.append(raw.strip())
        cleaned_text = ' '.join(parts)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = raw_text.strip() if isinstance(raw_text, str) else ''
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r'</?\w+[^>]*>', '', cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)

    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'

    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text


# ── K-core filtering ──────────────────────────────────────────────────

def _kcore_filter(interactions, user_core=5, item_core=5):
    """Iterative k-core filtering until convergence.

    Removes users with < user_core interactions and
    items with < item_core interactions, repeating until stable.
    Each interaction is a (user, item, timestamp, file_line_num) tuple.
    """
    print(f"  K-core filtering (user≥{user_core}, item≥{item_core})...")
    prev_len = -1
    while len(interactions) != prev_len:
        prev_len = len(interactions)

        # Filter items
        item_counts = collections.Counter(i for _, i, _, _ in interactions)
        interactions = [x for x in interactions if item_counts[x[1]] >= item_core]

        # Filter users
        user_counts = collections.Counter(u for u, _, _, _ in interactions)
        interactions = [x for x in interactions if user_counts[x[0]] >= user_core]

    n_users = len(set(u for u, _, _, _ in interactions))
    n_items = len(set(i for _, i, _, _ in interactions))
    print(f"  After filtering: {n_users} users, {n_items} items, {len(interactions)} interactions")
    return interactions


# ── Build inter.json + item.json ──────────────────────────────────────

def _build_output(interactions, metadata, out_dir: str, dataset: str):
    """Build inter.json and item.json with 0-indexed integer IDs.

    inter.json: {user_id(str): [item_id(int), ...]}  - chronological order
    item.json:  {item_id(str): {title, description}}

    Each interaction is (user, item, timestamp, file_line_num).
    """
    # P5/LETTER convention: group by user (file appearance order), sort within
    # user by timestamp, then assign IDs by iterating user-by-user.
    # This is equivalent to P5's make_inters_in_order() + sequential ID assignment.

    # Step A: determine user ordering by first file appearance (min line_num)
    user_order = {}  # raw_user → min file_line_num
    for user, _, _, ln in interactions:
        if user not in user_order or ln < user_order[user]:
            user_order[user] = ln

    # Step B: group by user, sort each user's items by timestamp
    user_groups = collections.defaultdict(list)
    for user, item, ts, ln in interactions:
        user_groups[user].append((ts, item))
    for user in user_groups:
        user_groups[user].sort()  # sort by timestamp

    # Step C: iterate users in file appearance order, assign IDs sequentially
    sorted_users = sorted(user_order.keys(), key=lambda u: user_order[u])
    user2id = {}
    item2id = {}
    for user in sorted_users:
        if user not in user2id:
            user2id[user] = len(user2id)
        for _, item in user_groups[user]:
            if item not in item2id:
                item2id[item] = len(item2id)

    # Build per-user sequences with integer IDs
    user_seqs = {}
    for user in sorted_users:
        uid = user2id[user]
        user_seqs[uid] = [(ts, item2id[item]) for ts, item in user_groups[user]]

    # inter.json: {user_id: [item_id, ...]}
    inter = {str(uid): [item_id for _, item_id in seq] for uid, seq in sorted(user_seqs.items())}

    # item.json: {item_id: {title, description, brand, categories}}
    # P5 convention: items with empty title/description get placeholder text
    item_json = {}
    placeholder_count = 0
    for asin, idx in item2id.items():
        meta = metadata.get(asin, {})
        entry = {
            "title": meta.get("title", ""),
            "description": meta.get("description", ""),
            "brand": meta.get("brand", ""),
            "categories": meta.get("categories", ""),
        }
        # P5 fallback: placeholder text for items with empty title+description
        if not entry["title"] and not entry["description"]:
            entry["title"] = f"title_{idx + 1}"
            entry["description"] = f"description_{idx + 1}"
            placeholder_count += 1
        item_json[str(idx)] = entry

    if placeholder_count > 0:
        print(f"  Placeholder text for {placeholder_count} items (empty title+desc)")

    # Save
    ds_dir = os.path.join(out_dir, dataset)
    os.makedirs(ds_dir, exist_ok=True)

    inter_path = os.path.join(ds_dir, f"{dataset}.inter.json")
    item_path = os.path.join(ds_dir, f"{dataset}.item.json")

    with open(inter_path, "w") as f:
        json.dump(inter, f)
    with open(item_path, "w") as f:
        json.dump(item_json, f, indent=2)

    # Statistics
    n_users = len(inter)
    n_items = len(item_json)
    n_inters = sum(len(v) for v in inter.values())
    seq_lens = [len(v) for v in inter.values()]
    print(f"\n  Dataset: {dataset}")
    print(f"  Users: {n_users}, Items: {n_items}, Interactions: {n_inters}")
    print(f"  Seq length: min={min(seq_lens)}, max={max(seq_lens)}, avg={n_inters/n_users:.2f}")
    print(f"  Saved: {inter_path}")
    print(f"  Saved: {item_path}")

    return inter_path, item_path


# ── Main pipeline ─────────────────────────────────────────────────────

def preprocess_amazon(dataset: str, version: str, raw_dir: str, out_dir: str,
                      user_core: int = 5, item_core: int = 5,
                      download: bool = True):
    """Full preprocessing pipeline: download → load → filter → save.

    Args:
        dataset: Short name (e.g. 'Beauty', 'Instruments').
        version: '2014' or '2018'.
        raw_dir: Directory for raw downloaded files.
        out_dir: Output directory for inter.json + item.json.
        user_core: Minimum interactions per user.
        item_core: Minimum interactions per item.
        download: Whether to download raw data (skip if already present).
    """
    print(f"{'='*60}")
    print(f"Preprocessing Amazon {version} — {dataset}")
    print(f"{'='*60}")

    # Step 1: Download
    if download:
        files = download_amazon(dataset, version, raw_dir)
    else:
        # Locate existing files
        if version == "2014":
            categories = AMAZON_2014_CATEGORIES
            full_name = categories[dataset]
            dest_dir = os.path.join(raw_dir, "amazon2014")
            files = {
                "reviews_file": os.path.join(dest_dir, f"reviews_{full_name}_5.json.gz"),
                "meta_file": os.path.join(dest_dir, f"meta_{full_name}.json.gz"),
            }
        else:
            categories = AMAZON_2018_CATEGORIES
            full_name = categories[dataset]
            dest_dir = os.path.join(raw_dir, "amazon2018")
            files = {
                "reviews_file": os.path.join(dest_dir, f"{full_name}_5.json.gz"),
                "meta_file": os.path.join(dest_dir, f"meta_{full_name}.json.gz"),
            }
        for k, v in files.items():
            if not os.path.exists(v):
                raise FileNotFoundError(f"Expected raw file not found: {v}")

    # Step 2: Load
    print("\nLoading raw data...")
    interactions = _load_reviews(files["reviews_file"], version)
    metadata = _load_metadata(files["meta_file"])

    # Metadata filter + deduplication (steps 3-4 below).
    # Applied uniformly to both 2014 and 2018 for consistent preprocessing.
    # Amazon 2014 _5.json.gz is already clean (no-op), while 2018 contains
    # items without metadata and duplicate user-item interactions (~5%).

    # Step 3: Filter by metadata (only keep items with metadata)
    meta_asins = set(metadata.keys())
    before = len(interactions)
    interactions = [x for x in interactions if x[1] in meta_asins]
    print(f"  After meta filter: {len(interactions)} (removed {before - len(interactions)})")

    # Step 4: Deduplicate (keep first interaction per user-item pair by timestamp)
    interactions.sort(key=lambda x: (x[0], x[2]))  # sort by user, then timestamp
    deduped = []
    seen = set()
    for u, i, t, ln in interactions:
        key = (u, i)
        if key not in seen:
            seen.add(key)
            deduped.append((u, i, t, ln))
    print(f"  After dedup: {len(deduped)} (removed {len(interactions) - len(deduped)})")
    interactions = deduped

    # Step 5: K-core filter
    interactions = _kcore_filter(interactions, user_core, item_core)

    # Step 6: Build and save
    print("\nBuilding output files...")
    _build_output(interactions, metadata, out_dir, dataset)

    print(f"\nDone: {dataset}")


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Amazon Review data")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset short name (e.g. Beauty, Instruments, Sports)")
    parser.add_argument("--version", type=str, required=True, choices=["2014", "2018"],
                        help="Amazon dataset version")
    parser.add_argument("--raw_dir", type=str, default="../raw_data",
                        help="Directory for raw downloaded files")
    parser.add_argument("--out_dir", type=str, default="../data",
                        help="Output directory for processed data")
    parser.add_argument("--user_core", type=int, default=5)
    parser.add_argument("--item_core", type=int, default=5)
    parser.add_argument("--no_download", action="store_true",
                        help="Skip download, use existing raw files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_amazon(
        dataset=args.dataset,
        version=args.version,
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        user_core=args.user_core,
        item_core=args.item_core,
        download=not args.no_download,
    )
