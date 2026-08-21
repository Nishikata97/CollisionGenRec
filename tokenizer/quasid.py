"""QuaSID-style RQ-VAE tokenizer (Hu et al., 2026).

Extends TIGER RQ-VAE with three components:
  - HaMR (Hamming-guided Margin Repulsion): severity-aware collision repulsion
  - CVPM (Conflict-Aware Valid Pair Masking): filters benign collision pairs
  - L_cl (Collaborative contrastive loss): InfoNCE on PPMI co-occurring pairs

Uses PPMI instead of Swing for controlled comparison with other tokenizers.

Reference: "Stop Treating Collisions Equally: Qualification-Aware Semantic ID
Learning for Recommendation at Industrial Scale" (arXiv 2603.00632)
"""

import collections
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset

from rqvae import (
    TigerRQVAE, _default_layers, _MLP, _VectorQuantizer, _ResidualVQ,
)


# ---------------------------------------------------------------------------
# Dataset (returns embedding + item index)
# ---------------------------------------------------------------------------

class _EmbIdxDataset(Dataset):
    def __init__(self, emb_path):
        self.data = np.load(emb_path).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), idx


# ---------------------------------------------------------------------------
# PPMI matrix construction (sparse)
# ---------------------------------------------------------------------------

def build_ppmi_sparse(interactions, n_items, window_size=3, holdout=2):
    """Build sparse PPMI matrix from interaction sequences."""
    rows, cols, vals = [], [], []
    for user, items in interactions.items():
        seq = items[:len(items) - holdout] if holdout > 0 else items
        for i in range(len(seq)):
            for j in range(i + 1, min(i + window_size + 1, len(seq))):
                a, b = seq[i], seq[j]
                if a < n_items and b < n_items:
                    rows.extend([a, b])
                    cols.extend([b, a])
                    vals.extend([1, 1])
    cooccur = csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))

    # PPMI transform
    total = cooccur.sum()
    if total == 0:
        return cooccur
    row_sum = np.array(cooccur.sum(axis=1)).flatten()
    col_sum = np.array(cooccur.sum(axis=0)).flatten()

    cooccur_coo = cooccur.tocoo()
    pmi_vals = []
    for r, c, v in zip(cooccur_coo.row, cooccur_coo.col, cooccur_coo.data):
        pmi = np.log(v * total / (row_sum[r] * col_sum[c] + 1e-10) + 1e-10)
        pmi_vals.append(max(pmi, 0))  # positive PMI only
    ppmi = csr_matrix((pmi_vals, (cooccur_coo.row, cooccur_coo.col)),
                      shape=(n_items, n_items))
    return ppmi


# ---------------------------------------------------------------------------
# QuaSID losses
# ---------------------------------------------------------------------------

def compute_hamr_loss(z, indices, cvpm_mask, m_full=0.8, m_partial=0.5,
                      R=1, lambda_full=0.2, lambda_partial=0.1):
    """Hamming-guided Margin Repulsion (Eq.9-13 in QuaSID)."""
    B = z.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z.device)

    # Pairwise Hamming distance (B, B)
    hamming = (indices.unsqueeze(0) != indices.unsqueeze(1)).sum(dim=-1).float()

    # Pairwise cosine distance in encoder space (Eq.10 in QuaSID)
    z_norm = F.normalize(z, dim=1)
    z_dist = 1.0 - (z_norm @ z_norm.t())  # cosine distance ∈ [0, 2]

    # Exclude diagonal
    diag_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    cvpm_bool = (cvpm_mask > 0) & diag_mask

    # Full collision: H=0
    full_mask = (hamming == 0) & cvpm_bool
    if full_mask.any():
        loss_full = F.relu(m_full - z_dist[full_mask]).mean()
    else:
        loss_full = torch.tensor(0.0, device=z.device)

    # Partial collision: 0 < H <= R
    partial_mask = (hamming > 0) & (hamming <= R) & cvpm_bool
    if partial_mask.any():
        loss_partial = F.relu(m_partial - z_dist[partial_mask]).mean()
    else:
        loss_partial = torch.tensor(0.0, device=z.device)

    return lambda_full * loss_full + lambda_partial * loss_partial


def build_cvpm_mask(batch_idx, ppmi_matrix, device):
    """Conflict-Aware Valid Pair Masking (Eq.8 in QuaSID).

    Returns (B, B) mask where 1 = qualified collision pair (repel allowed).
    """
    B = len(batch_idx)
    idx = batch_idx.cpu().numpy() if torch.is_tensor(batch_idx) else np.array(batch_idx)

    # Mask 1: collaborative-positive exclusion
    # If PPMI(i,j) > 0 → these are co-occurring pairs → don't repel
    ppmi_sub = ppmi_matrix[idx][:, idx]  # (B, B) sparse submatrix
    if hasattr(ppmi_sub, 'toarray'):
        ppmi_dense = ppmi_sub.toarray()
    else:
        ppmi_dense = np.array(ppmi_sub)
    collab_mask = (ppmi_dense == 0).astype(np.float32)  # 1 = no co-occurrence → repel OK

    # Mask 2: same-item exclusion (in-batch duplicates)
    same_mask = (idx[:, None] != idx[None, :]).astype(np.float32)

    # Hadamard product
    mask = collab_mask * same_mask
    return torch.tensor(mask, dtype=torch.float32, device=device)


def compute_cl_loss(z, batch_idx, ppmi_matrix, temperature=0.1):
    """Collaborative contrastive loss (InfoNCE on PPMI pairs)."""
    B = z.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z.device)

    idx = batch_idx.cpu().numpy() if torch.is_tensor(batch_idx) else np.array(batch_idx)

    # Find positive pairs: PPMI(i,j) > 0 within the batch
    ppmi_sub = ppmi_matrix[idx][:, idx]
    if hasattr(ppmi_sub, 'toarray'):
        ppmi_dense = ppmi_sub.toarray()
    else:
        ppmi_dense = np.array(ppmi_sub)

    # For each item, find its positive (highest PPMI in batch)
    np.fill_diagonal(ppmi_dense, 0)
    has_positive = ppmi_dense.max(axis=1) > 0
    if not has_positive.any():
        return torch.tensor(0.0, device=z.device)

    positives = ppmi_dense.argmax(axis=1)  # (B,) best positive per item

    # Cosine similarity matrix
    z_norm = F.normalize(z, dim=1)
    sim = z_norm @ z_norm.t() / temperature  # (B, B)

    # False-negative masking (Eq.7): exclude collaborative positives
    # (other than THE positive) from denominator by setting sim to -inf
    fn_mask = torch.tensor(ppmi_dense > 0, dtype=torch.bool, device=z.device)
    # Keep the selected positive visible
    for i in range(B):
        if has_positive[i]:
            fn_mask[i, positives[i]] = False
    # Also keep self visible (diagonal) - cross_entropy handles it
    fn_mask.fill_diagonal_(False)
    sim = sim.masked_fill(fn_mask, -1e9)

    # InfoNCE: for items that have a positive, compute cross-entropy
    mask = torch.tensor(has_positive, device=z.device)
    if not mask.any():
        return torch.tensor(0.0, device=z.device)

    targets = torch.tensor(positives, dtype=torch.long, device=z.device)
    loss = F.cross_entropy(sim[mask], targets[mask])
    return loss


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class _QuaSIDTrainer:
    def __init__(self, model, lr, weight_decay, epochs, eval_step, patience,
                 device, ckpt_dir, ppmi_matrix,
                 m_full=0.8, m_partial=0.5, R=1,
                 lambda_full=0.2, lambda_partial=0.1, lambda_cl=0.1):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.epochs = epochs
        self.eval_step = eval_step
        self.patience = patience
        self.ppmi_matrix = ppmi_matrix
        self.m_full = m_full
        self.m_partial = m_partial
        self.R = R
        self.lambda_full = lambda_full
        self.lambda_partial = lambda_partial
        self.lambda_cl = lambda_cl

        import time
        ts = time.strftime("%b-%d-%Y_%H-%M-%S")
        self.ckpt_dir = os.path.join(ckpt_dir, ts)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        self.best_collision_rate = float("inf")
        self.logger = logging.getLogger(__name__)

    def _save(self, epoch, collision_rate, filename=None):
        path = filename or os.path.join(
            self.ckpt_dir,
            f"epoch_{epoch}_collision_{collision_rate:.4f}_model.pth")
        torch.save({"state_dict": self.model.state_dict(),
                     "epoch": epoch,
                     "best_collision_rate": self.best_collision_rate},
                   path)

    @torch.no_grad()
    def _eval_collision(self, loader):
        """Collision rate: items_in_non_singleton_groups / n_items."""
        self.model.eval()
        all_codes = []
        for batch, _ in loader:
            batch = batch.to(self.device)
            indices = self.model.get_indices(batch)
            all_codes.extend(indices.cpu().tolist())
        from collections import Counter
        code_strs = [str(c) for c in all_codes]
        n = len(code_strs)
        if n == 0:
            return 0.0
        group_sizes = Counter(code_strs)
        items_in_collision = sum(s for s in group_sizes.values() if s > 1)
        return items_in_collision / n

    def fit(self, loader):
        best_loss = float("inf")
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for batch, batch_idx in loader:
                batch = batch.to(self.device)
                out, rq_loss, indices = self.model(batch)
                loss_total, loss_recon = self.model.compute_loss(
                    out, rq_loss, batch)

                # Encoder output for HaMR and CL
                z = self.model.encoder(batch)

                # CVPM mask
                cvpm_mask = build_cvpm_mask(
                    batch_idx, self.ppmi_matrix, self.device)

                # HaMR loss
                loss_hamr = compute_hamr_loss(
                    z, indices, cvpm_mask,
                    m_full=self.m_full, m_partial=self.m_partial,
                    R=self.R, lambda_full=self.lambda_full,
                    lambda_partial=self.lambda_partial)

                # Collaborative contrastive loss
                loss_cl = compute_cl_loss(
                    z, batch_idx, self.ppmi_matrix)

                # Total loss
                loss = loss_total + loss_hamr + self.lambda_cl * loss_cl

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % self.eval_step == 0:
                coll = self._eval_collision(loader)
                avg_loss = epoch_loss / len(loader)
                self.logger.info(
                    f"epoch {epoch}: loss={avg_loss:.4f} collision={coll:.4f}")
                self._save(epoch, coll)
                if coll < self.best_collision_rate:
                    self.best_collision_rate = coll
                    self._save(epoch, coll,
                               os.path.join(self.ckpt_dir,
                                            "best_collision_model.pth"))
                if avg_loss < best_loss:
                    best_loss = avg_loss

        return best_loss, self.best_collision_rate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_quasid(
    embeddings: np.ndarray,
    ckpt_dir: str,
    interactions: dict = None,
    n_layers: int = 4,
    codebook_size: int = 256,
    e_dim: int = 32,
    epochs: int = 20000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: str = "cuda:0",
    eval_step: int = 2000,
    patience: int = 20000,
    zcr: bool = False,
    zcr_mode: str = "greedy",
    # QuaSID-specific
    m_full: float = 0.8,
    m_partial: float = 0.5,
    R: int = 1,
    lambda_full: float = 0.2,
    lambda_partial: float = 0.1,
    lambda_cl: float = 0.1,
    window_size: int = 3,
    holdout: int = 2,
    verbose: bool = True,
) -> dict:
    """Train QuaSID-style RQ-VAE and return index dict."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    n_items, in_dim = embeddings.shape
    os.makedirs(ckpt_dir, exist_ok=True)

    layers = _default_layers(in_dim, e_dim)
    if verbose:
        print(f"  QuaSID RQ-VAE: {n_items} items, {in_dim}d -> {layers} -> {e_dim}d")

    # Build PPMI matrix
    if interactions is not None:
        if verbose:
            print(f"  Building PPMI matrix (window={window_size}, holdout={holdout})...")
        ppmi_matrix = build_ppmi_sparse(
            interactions, n_items, window_size=window_size, holdout=holdout)
        if verbose:
            print(f"  PPMI: {ppmi_matrix.nnz} nonzero entries")
    else:
        ppmi_matrix = csr_matrix((n_items, n_items))

    emb_path = os.path.join(ckpt_dir, "rqvae_input_emb.npy")
    np.save(emb_path, embeddings)

    model = TigerRQVAE(
        in_dim=in_dim,
        num_emb_list=[codebook_size] * n_layers,
        e_dim=e_dim,
        layers=layers,
        beta=0.25,
        kmeans_init=True,
        kmeans_iters=100,
    )

    dataset = _EmbIdxDataset(emb_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    logging.basicConfig(level=logging.INFO)
    trainer = _QuaSIDTrainer(
        model, lr=lr, weight_decay=1e-4, epochs=epochs,
        eval_step=eval_step, patience=patience,
        device=device, ckpt_dir=ckpt_dir,
        ppmi_matrix=ppmi_matrix,
        m_full=m_full, m_partial=m_partial, R=R,
        lambda_full=lambda_full, lambda_partial=lambda_partial,
        lambda_cl=lambda_cl,
    )
    _, best_coll = trainer.fit(loader)
    if verbose:
        print(f"  Training done. best_collision={best_coll:.4f}")

    # Load best checkpoint
    dev = torch.device(device)
    best_path = os.path.join(trainer.ckpt_dir, "best_collision_model.pth")
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=dev, weights_only=False)
        model.load_state_dict(state["state_dict"])
        if verbose:
            print(f"  Loaded: {best_path}")
    model.eval()
    model.to(dev)

    # Generate indices
    all_emb = torch.tensor(embeddings, dtype=torch.float32)
    all_indices = []
    with torch.no_grad():
        for start in range(0, n_items, batch_size):
            batch = all_emb[start:start + batch_size].to(dev)
            all_indices.append(model.get_indices(batch).cpu())
    all_indices = torch.cat(all_indices, dim=0)

    # Optional ZCR
    if zcr:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from zcr import apply_zcr

        with torch.no_grad():
            latents = []
            for start in range(0, n_items, batch_size):
                batch = all_emb[start:start + batch_size].to(dev)
                latents.append(model.encoder(batch).cpu())
            latents = torch.cat(latents, dim=0).numpy()

        centroids = [vq.embedding.weight.cpu().detach()
                     for vq in model.rq.vq_layers]
        codes_np, zcr_stats = apply_zcr(
            all_indices.numpy(), centroids, latents, codebook_size,
            mode=zcr_mode)
        if verbose and zcr_stats["n_reassigned"] > 0:
            print(f"  ZCR: {zcr_stats['n_reassigned']} items reassigned")
    else:
        codes_np = all_indices.numpy()
        zcr_stats = {}

    # Build index dict
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>", "<f_{}>"]
    index_dict = {}
    for i in range(n_items):
        index_dict[i] = [prefix[l].format(int(codes_np[i, l]))
                         for l in range(n_layers)]

    n_unique = len(set(tuple(v) for v in index_dict.values()))
    coll = (n_items - n_unique) / n_items
    if verbose:
        print(f"  Final: {n_unique}/{n_items} unique, collision={coll:.4f}")

    # Return tuple (index_dict, codes_after, zcr_stats) for sidecar writing.
    return index_dict, codes_np, zcr_stats
