#!/usr/bin/env python
"""Build a Semantic ID (SID) index for one dataset.

Pipeline:
  1. Load textual item embeddings
  2. (optional) Extract collaborative signal (PPMI+SVD) and fuse via concat + PCA
  3. Quantize (RK-Means, RQ-VAE, LETTER, or QuaSID)
  4. (optional) Zero-Collision Reassignment (ZCR)

Usage:
    # RK-Means (native)
    python tokenizer/build_index.py --dataset Beauty --alpha 0.0

    # RK-Means + ZCR
    python tokenizer/build_index.py --dataset Beauty --alpha 0.0 --zcr

    # RK-Means + collaborative fusion + ZCR
    python tokenizer/build_index.py --dataset Beauty --alpha 0.5 --zcr

    # RQ-VAE (TIGER-style), native / + ZCR
    python tokenizer/build_index.py --dataset Beauty --quantizer rqvae
    python tokenizer/build_index.py --dataset Beauty --quantizer rqvae --zcr
"""

import argparse
import json
import os

import random

import numpy as np
import torch

from collaborative_signal import CollaborativeSignal
from zcr import apply_zcr
from fusion import fuse_embeddings
from residual_kmeans import ResKmeans
import rqvae
import letter
import quasid


def generate_index(codes_np: np.ndarray, dataset: str,
                   output_dir: str, suffix: str) -> str:
    """Convert codes array to index JSON."""
    prefix_fmt = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>", "<f_{}>"]
    index_dict = {}
    for item_id in range(len(codes_np)):
        tokens = [prefix_fmt[l].format(int(codes_np[item_id, l]))
                  for l in range(codes_np.shape[1])]
        index_dict[item_id] = tokens

    # Collision stats: items in non-singleton groups / total items
    from collections import Counter
    code_strs = [str(c.tolist()) for c in codes_np]
    group_sizes = Counter(code_strs)
    n_unique = len(group_sizes)
    n_items = len(codes_np)
    items_in_collision = sum(s for s in group_sizes.values() if s > 1)
    collision_rate = items_in_collision / n_items if n_items > 0 else 0.0
    g_max = max(group_sizes.values()) if group_sizes else 0
    print(f"  Items: {n_items}, Unique: {n_unique}, "
          f"Collision rate: {collision_rate:.4f} (G_max: {g_max})")

    index_dir = os.path.join(output_dir, "indexes", suffix)
    os.makedirs(index_dir, exist_ok=True)
    output_file = os.path.join(index_dir, f"{dataset}.index.json")
    with open(output_file, "w") as f:
        json.dump(index_dict, f)
    print(f"  Index saved to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Collaborative Semantic ID tokenizer")

    # Paths
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--emb_path", type=str, default=None,
                        help="Semantic embeddings .npy (default: auto)")
    parser.add_argument("--inter_path", type=str, default=None,
                        help="Interaction data .json (default: auto)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)

    # Fusion
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="CF weight (0=pure RK-Means)")
    parser.add_argument("--cs_method", type=str, default="ppmi_svd",
                        choices=["ppmi_svd"])
    parser.add_argument("--d_cf", type=int, default=256)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--holdout", type=int, default=2)

    # Quantization
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--niter", type=int, default=20)
    parser.add_argument("--quantizer", type=str, default="rkmeans",
                        choices=["rkmeans", "rqvae", "letter", "quasid"])

    # RQ-VAE specific
    parser.add_argument("--rqvae_epochs", type=int, default=20000)
    parser.add_argument("--rqvae_e_dim", type=int, default=32)
    parser.add_argument("--rqvae_batch_size", type=int, default=8192)
    parser.add_argument("--rqvae_lr", type=float, default=1e-3)
    parser.add_argument("--rqvae_eval_step", type=int, default=2000)
    parser.add_argument("--rqvae_patience", type=int, default=5)
    parser.add_argument("--rqvae_device", type=str, default="cuda:0")

    # Zero-Collision Reassignment (ZCR)
    parser.add_argument("--zcr", action="store_true",
                        help="Last-level ZCR for zero collision")
    parser.add_argument("--zcr_mode", type=str, default="optimal",
                        choices=["greedy", "optimal"],
                        help="ZCR algorithm: 'optimal' (default, ZCR via "
                             "min-cost bipartite matching) or 'greedy' "
                             "(greedy reassignment baseline). Only effective when --zcr "
                             "is set. Suffix: optimal → '_zcr', greedy → '_greedy_zcr'.")

    # Embedding tag (for distinguishing different embedding sources in output filenames)
    parser.add_argument("--emb_tag", type=str, default=None,
                        help="Tag added to index filename (e.g. 'td' for title+description)")

    # Canonical variant name override (used by scripts/tokenize.sh)
    parser.add_argument("--variant_name", type=str, default=None,
                        help="Override auto-generated suffix with explicit canonical name "
                             "(e.g. 'rkmeans_zcr', 'rqvae_zcr'). When provided, index_dir = "
                             "<output_dir>/indexes/<variant_name> regardless of zcr_tag/quantizer.")

    args = parser.parse_args()

    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Resolve defaults
    if args.emb_path is None:
        args.emb_path = f"./data/{args.dataset}/{args.dataset}.emb-qwen3.npy"
    if args.inter_path is None:
        args.inter_path = f"./data/{args.dataset}/{args.dataset}.inter.json"
    if args.output_dir is None:
        args.output_dir = f"./data/{args.dataset}"
    if args.ckpt_dir is None:
        args.ckpt_dir = f"./checkpoint/{args.dataset}"

    # Build suffix
    alpha_str = str(args.alpha).replace(".", "p")
    v_tag = f"_v{args.codebook_size}" if args.codebook_size != 256 else ""
    l_tag = f"_l{args.n_layers}" if args.n_layers != 4 else ""
    if not args.zcr:
        zcr_tag = ""
    elif args.zcr_mode == "greedy":
        zcr_tag = "_greedy_zcr"
    else:  # optimal is now the default convention
        zcr_tag = "_zcr"
    e_tag = f"_{args.emb_tag}" if args.emb_tag else ""

    if args.variant_name:
        # Canonical-name override
        suffix = args.variant_name
    elif args.quantizer == "rqvae":
        suffix = f"rqvae_a{alpha_str}{e_tag}{v_tag}{l_tag}{zcr_tag}"
    elif args.quantizer == "letter":
        suffix = f"letter_a{alpha_str}{e_tag}{v_tag}{l_tag}{zcr_tag}"
    elif args.quantizer == "quasid":
        suffix = f"quasid_a{alpha_str}{e_tag}{v_tag}{l_tag}{zcr_tag}"
    elif args.alpha == 0.0:
        suffix = f"sid_a0p0{e_tag}{v_tag}{l_tag}{zcr_tag}"
    else:
        d_tag = f"_d{args.d_cf}" if args.d_cf != 256 else ""
        suffix = f"sid_a{alpha_str}{d_tag}{e_tag}{v_tag}{l_tag}{zcr_tag}"

    # ── 1. Load semantic embeddings ──
    print(f"Loading semantic embeddings from {args.emb_path}")
    E_sem = np.load(args.emb_path).astype(np.float32)
    n_items = E_sem.shape[0]
    print(f"  Shape: {E_sem.shape}")

    # ── 2. Collaborative signal ──
    E_cf = None
    if args.alpha > 0 and args.quantizer != "rqvae":
        print(f"\nComputing collaborative signal (method={args.cs_method})...")
        with open(args.inter_path) as f:
            interactions = json.load(f)
        print(f"  Users: {len(interactions)}")
        cs = CollaborativeSignal(
            method=args.cs_method, window_size=args.window_size,
            d_cf=args.d_cf, holdout=args.holdout,
        )
        E_cf = cs.fit(interactions, n_items)

    # ── 3. Quantize ──
    if args.quantizer == "rqvae":
        E_input = E_sem
        rqvae_ckpt = os.path.join(args.ckpt_dir, f"{suffix}")

        print(f"\nTraining rqvae (epochs={args.rqvae_epochs})...")
        index_dict, codes_np, zcr_stats = rqvae.train_rqvae(
            embeddings=E_input, ckpt_dir=rqvae_ckpt,
            n_layers=args.n_layers, codebook_size=args.codebook_size,
            e_dim=args.rqvae_e_dim, epochs=args.rqvae_epochs,
            batch_size=args.rqvae_batch_size, lr=args.rqvae_lr,
            device=args.rqvae_device, eval_step=args.rqvae_eval_step,
            patience=args.rqvae_patience,
            zcr=args.zcr,
            zcr_mode=args.zcr_mode,
        )

        index_dir = os.path.join(args.output_dir, "indexes", suffix)
        os.makedirs(index_dir, exist_ok=True)
        output_file = os.path.join(index_dir, f"{args.dataset}.index.json")
        with open(output_file, "w") as f:
            json.dump(index_dict, f)
        print(f"\nIndex saved to {output_file}")

        # ZCR provenance + stats sidecars
        from zcr_provenance import write_zcr_sidecars
        write_zcr_sidecars(
            index_dir=index_dir, dataset=args.dataset,
            zcr_mode=(args.zcr_mode if args.zcr else "none"),
            zcr_stats=zcr_stats, codes_after=codes_np,
            codebook_size=args.codebook_size,
        )

    elif args.quantizer == "letter":
        E_input = E_sem
        rqvae_ckpt = os.path.join(args.ckpt_dir, f"{suffix}")

        with open(args.inter_path) as f:
            interactions = json.load(f)
        print(f"\nTraining letter (epochs={args.rqvae_epochs})...")
        index_dict, codes_np, zcr_stats = letter.train_rqvae(
            embeddings=E_input, ckpt_dir=rqvae_ckpt,
            interactions=interactions,
            n_layers=args.n_layers, codebook_size=args.codebook_size,
            e_dim=args.rqvae_e_dim, epochs=args.rqvae_epochs,
            batch_size=args.rqvae_batch_size, lr=args.rqvae_lr,
            device=args.rqvae_device, eval_step=args.rqvae_eval_step,
            patience=args.rqvae_patience,
            zcr=args.zcr,
            zcr_mode=args.zcr_mode,
        )

        index_dir = os.path.join(args.output_dir, "indexes", suffix)
        os.makedirs(index_dir, exist_ok=True)
        output_file = os.path.join(index_dir, f"{args.dataset}.index.json")
        with open(output_file, "w") as f:
            json.dump(index_dict, f)
        print(f"\nIndex saved to {output_file}")

        # ZCR provenance + stats sidecars
        from zcr_provenance import write_zcr_sidecars
        write_zcr_sidecars(
            index_dir=index_dir, dataset=args.dataset,
            zcr_mode=(args.zcr_mode if args.zcr else "none"),
            zcr_stats=zcr_stats, codes_after=codes_np,
            codebook_size=args.codebook_size,
        )

    elif args.quantizer == "quasid":
        E_input = E_sem
        rqvae_ckpt = os.path.join(args.ckpt_dir, f"{suffix}")

        with open(args.inter_path) as f:
            interactions = json.load(f)
        print(f"\nTraining quasid (epochs={args.rqvae_epochs})...")
        index_dict, codes_np, zcr_stats = quasid.train_quasid(
            embeddings=E_input, ckpt_dir=rqvae_ckpt,
            interactions=interactions,
            n_layers=args.n_layers, codebook_size=args.codebook_size,
            e_dim=args.rqvae_e_dim, epochs=args.rqvae_epochs,
            batch_size=args.rqvae_batch_size, lr=args.rqvae_lr,
            device=args.rqvae_device, eval_step=args.rqvae_eval_step,
            patience=args.rqvae_patience,
            zcr=args.zcr,
            zcr_mode=args.zcr_mode,
        )

        index_dir = os.path.join(args.output_dir, "indexes", suffix)
        os.makedirs(index_dir, exist_ok=True)
        output_file = os.path.join(index_dir, f"{args.dataset}.index.json")
        with open(output_file, "w") as f:
            json.dump(index_dict, f)
        print(f"\nIndex saved to {output_file}")

        # ZCR provenance + stats sidecars
        from zcr_provenance import write_zcr_sidecars
        write_zcr_sidecars(
            index_dir=index_dir, dataset=args.dataset,
            zcr_mode=(args.zcr_mode if args.zcr else "none"),
            zcr_stats=zcr_stats, codes_after=codes_np,
            codebook_size=args.codebook_size,
        )

    else:
        # RK-Means path
        if args.alpha > 0 and E_cf is not None:
            print(f"\nFusing embeddings (alpha={args.alpha})...")
            E_fused = fuse_embeddings(E_sem, E_cf, args.alpha)
        else:
            print("\nalpha=0.0, using raw semantic embeddings (standard RK-Means)")
            E_fused = E_sem.copy()

        dim = E_fused.shape[1]
        print(f"\nTraining RK-Means: n_layers={args.n_layers}, "
              f"V={args.codebook_size}, dim={dim}")

        model = ResKmeans(args.n_layers, args.codebook_size, dim)
        model.train_kmeans(torch.tensor(E_fused), niter=args.niter, verbose=True)

        # Save checkpoint
        os.makedirs(args.ckpt_dir, exist_ok=True)
        model_path = os.path.join(args.ckpt_dir, f"rkmeans_{suffix}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")

        # Encode
        with torch.no_grad():
            codes = model.encode(torch.tensor(E_fused))

        # Zero-Collision Reassignment (ZCR)
        if args.zcr:
            centroids = [model.centroids[l].detach() for l in range(model.n_layers)]
            codes_np, zcr_stats = apply_zcr(
                codes, centroids, E_fused, model.codebook_size,
                mode=args.zcr_mode)
            print(f"  ZCR: {zcr_stats['n_reassigned']} items reassigned "
                  f"across {zcr_stats['n_collision_groups']} groups")
            print(f"    Avg sq-dist increase: {zcr_stats['avg_dist_increase']:.4f}")
        else:
            codes_np = codes.numpy()

        # Generate index
        print(f"\nGenerating index...")
        generate_index(codes_np, args.dataset, args.output_dir, suffix)

        # Write ZCR provenance + stats sidecars (RK-Means branch only;
        # RQ-VAE family modules write their own sidecars internally.)
        from zcr_provenance import write_zcr_sidecars
        index_dir = os.path.join(args.output_dir, "indexes", suffix)
        zcr_mode_for_sidecar = args.zcr_mode if args.zcr else "none"
        zcr_stats_for_sidecar = zcr_stats if args.zcr else {}
        write_zcr_sidecars(
            index_dir=index_dir,
            dataset=args.dataset,
            zcr_mode=zcr_mode_for_sidecar,
            zcr_stats=zcr_stats_for_sidecar,
            codes_after=codes_np,
            codebook_size=args.codebook_size,
        )
        print(f"  Provenance sidecars written: zcr_mode={zcr_mode_for_sidecar}")

    print("\nDone!")


if __name__ == "__main__":
    main()
