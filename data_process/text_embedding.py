"""Generate item text embeddings.

Matches LETTER's field-averaging approach: embed each field separately,
then average field embeddings (equal weight per field).

Pipeline: item.json → clean_text() → PLM (per-field mean pooling) → field average → .npy

Usage:
    python text_embedding.py --dataset Beauty --data_dir ../data \
        --plm_path /path/to/Qwen3-Embedding-8B --plm_name qwen3

    # title+description only (matching LETTER convention):
    python text_embedding.py --dataset Beauty --data_dir ../data \
        --plm_path /path/to/Qwen3-Embedding-8B --plm_name qwen3 \
        --fields title,description

Output: {data_dir}/{dataset}/{dataset}.emb-{plm_name}[-{field_tag}].npy
    Shape: (n_items, hidden_dim), ordered by item_id (0, 1, ..., n_items-1)
"""

import argparse
import html
import json
import os
import re

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def clean_text(raw_text: str) -> str:
    """Clean text matching LETTER's utils.clean_text() exactly.

    Steps: html.unescape → strip HTML tags → remove "\\n\\r →
           trailing period normalization → truncate if ≥2000 chars.
    """
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
            cleaned_text = raw_text.strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r'</?\w+[^>]*>', '', cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)

    # Trailing period normalization
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


def load_item_texts(data_dir: str, dataset: str, fields: list = None):
    """Load per-field item texts from item.json.

    Returns list of per-field text lists (matching LETTER's generate_text),
    with clean_text() applied to each field.

    Returns:
        list of list[str]: texts[i] = [field1_text, field2_text, ...]
            for item i, ordered by item_id (0 .. n_items-1).
    """
    item_path = os.path.join(data_dir, dataset, f"{dataset}.item.json")
    with open(item_path, "r") as f:
        items = json.load(f)

    if fields is None:
        fields = ["title", "description", "brand", "categories"]
    n_items = len(items)
    # Each item gets a list of per-field cleaned texts
    texts = [None] * n_items
    field_usage = {f: 0 for f in fields}

    for item_id_str, meta in items.items():
        idx = int(item_id_str)
        field_texts = []
        for field in fields:
            raw_val = meta.get(field, "")
            cleaned = clean_text(raw_val)
            field_texts.append(cleaned.strip())
            if cleaned.strip():
                field_usage[field] += 1
        texts[idx] = field_texts

    # Check for missing items
    missing = sum(1 for t in texts if t is None)
    if missing > 0:
        print(f"  Warning: {missing}/{n_items} items missing from item.json")
        for i in range(n_items):
            if texts[i] is None:
                texts[i] = [""] * len(fields)

    usage_str = ", ".join(f"{f}={field_usage[f]}" for f in fields)
    print(f"  Loaded {n_items} item texts from {item_path}")
    print(f"  Fields: {fields}")
    print(f"  Field usage: {usage_str}")
    return texts


def generate_embeddings(texts: list, tokenizer, model, device,
                        batch_size: int = 1, max_length: int = 2048):
    """Generate embeddings via per-field mean pooling + field averaging.

    Matches LETTER's generate_item_embedding(): each field is embedded
    separately via mean pooling, then field embeddings are averaged
    with equal weight.

    Args:
        texts: list of list[str], texts[i] = [field1, field2, ...].
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace model.
        device: torch device.
        batch_size: inference batch size.
        max_length: max tokenization length.

    Returns:
        np.ndarray of shape (n_items, hidden_dim).
    """
    model.eval()
    all_embeddings = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_items = texts[start:start + batch_size]
        # batch_items: list of [field1, field2, ...] per item

        # Transpose: group by field across items in this batch
        # field_groups[f] = [item0_field_f, item1_field_f, ...]
        n_fields = len(batch_items[0])
        field_groups = list(zip(*batch_items))

        field_embeddings = []
        for field_texts in field_groups:
            sentences = list(field_texts)
            # Replace empty strings with a space to avoid tokenizer issues
            sentences = [s if s else " " for s in sentences]

            encoded = tokenizer(
                sentences,
                max_length=max_length,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                )

            # Mean pooling (matching LETTER: native dtype, no cast)
            masked_output = outputs.last_hidden_state * encoded.attention_mask.unsqueeze(-1)
            pooled = masked_output.sum(dim=1) / encoded.attention_mask.sum(dim=-1, keepdim=True)

            field_embeddings.append(pooled.cpu())

        # Average across fields (equal weight per field)
        field_mean = torch.stack(field_embeddings, dim=0).mean(dim=0)
        all_embeddings.append(field_mean)

    return torch.cat(all_embeddings, dim=0).numpy()


def main(args):
    print("=" * 60)
    print(f"Generating text embeddings: {args.dataset}")
    print(f"PLM: {args.plm_path} (name: {args.plm_name})")
    print(f"Method: per-field mean pooling + field averaging (LETTER-compatible)")
    print("=" * 60)

    # Load item texts
    fields = args.fields.split(",") if args.fields else None
    print(f"\nLoading item texts (fields={fields or 'all'})...")
    texts = load_item_texts(args.data_dir, args.dataset, fields=fields)

    # Load PLM
    print(f"\nLoading model: {args.plm_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    model = AutoModel.from_pretrained(
        args.plm_path,
        low_cpu_mem_usage=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}")
    print(f"  Model hidden size: {model.config.hidden_size}")

    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = generate_embeddings(
        texts, tokenizer, model, device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Save - include field suffix if not using all fields
    if fields and set(fields) != {"title", "description", "brand", "categories"}:
        field_tag = "".join(f[0] for f in fields)  # e.g. "td" for title,description
        emb_name = f"{args.dataset}.emb-{args.plm_name}-{field_tag}.npy"
    else:
        emb_name = f"{args.dataset}.emb-{args.plm_name}.npy"
    out_path = os.path.join(args.data_dir, args.dataset, emb_name)
    np.save(out_path, embeddings.astype(np.float32))
    print(f"\n  Saved: {out_path}")
    print(f"  Shape: {embeddings.shape}, dtype: float32")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate item text embeddings")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. Beauty, Instruments, Yelp)")
    parser.add_argument("--data_dir", type=str, default="../data",
                        help="Data directory containing {dataset}/{dataset}.item.json")
    parser.add_argument("--plm_path", type=str, required=True,
                        help="Path to pretrained language model")
    parser.add_argument("--plm_name", type=str, default="qwen3",
                        help="Short name for output file (default: qwen3)")
    parser.add_argument("--fields", type=str, default=None,
                        help="Comma-separated fields (default: all). E.g. 'title,description'")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Inference batch size (default: 1, increase if GPU memory allows)")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max tokenization length")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
