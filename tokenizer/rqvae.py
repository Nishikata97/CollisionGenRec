"""Standalone TIGER-style RQ-VAE tokenizer.

Standard VQ-VAE loss (reconstruction + codebook + commitment), no LETTER
additions (no diversity loss, no Sinkhorn, no constrained K-means).
Follows Rajput et al. (NeurIPS 2023) "Recommender Systems with Generative
Retrieval", verified against snap-research/GRID and EdoardoBotta/RQ-VAE-
Recommender implementations.

Input:  raw embeddings (n_items, d)
Output: {item_id (int): ["<a_X>", "<b_Y>", ...]} index dict
"""

import collections
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _EmbDataset(Dataset):
    def __init__(self, emb_path):
        self.data = np.load(emb_path).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), idx


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, dims, dropout=0.0):
        super().__init__()
        layers = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(d_in, d_out))
            if i != len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.net(x)


class _VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25, kmeans_init=True, kmeans_iters=10):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.embedding = nn.Embedding(n_e, e_dim)
        if not kmeans_init:
            self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)
            self.initted = True
        else:
            self.embedding.weight.data.zero_()
            self.initted = False

    def _kmeans_init(self, data):
        from sklearn.cluster import KMeans
        x = data.cpu().detach().numpy()
        clf = KMeans(n_clusters=self.n_e, max_iter=self.kmeans_iters, n_init=10)
        clf.fit(x)
        self.embedding.weight.data.copy_(
            torch.from_numpy(clf.cluster_centers_).to(data.device))
        self.initted = True

    def forward(self, x):
        flat = x.view(-1, self.e_dim)
        if not self.initted and self.training:
            self._kmeans_init(flat)

        d = (flat ** 2).sum(1, keepdim=True) \
            + (self.embedding.weight ** 2).sum(1).unsqueeze(0) \
            - 2 * flat @ self.embedding.weight.t()
        indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        x_q = x + (x_q - x).detach()  # STE
        return x_q, loss, indices.view(x.shape[:-1])


class _ResidualVQ(nn.Module):
    def __init__(self, n_e_list, e_dim, beta=0.25,
                 kmeans_init=True, kmeans_iters=10):
        super().__init__()
        self.vq_layers = nn.ModuleList([
            _VectorQuantizer(n_e, e_dim, beta=beta,
                             kmeans_init=kmeans_init, kmeans_iters=kmeans_iters)
            for n_e in n_e_list
        ])

    def forward(self, x):
        all_losses, all_indices = [], []
        residual = x
        x_q = 0
        for vq in self.vq_layers:
            x_res, loss, indices = vq(residual)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(indices)
        return x_q, torch.stack(all_losses).mean(), torch.stack(all_indices, dim=-1)


class TigerRQVAE(nn.Module):
    def __init__(self, in_dim, num_emb_list, e_dim, layers,
                 dropout_prob=0.0, loss_type="mse", quant_loss_weight=1.0,
                 beta=0.25, kmeans_init=True, kmeans_iters=10):
        super().__init__()
        self.quant_loss_weight = quant_loss_weight
        self.loss_type = loss_type
        enc_dims = [in_dim] + layers + [e_dim]
        self.encoder = _MLP(enc_dims, dropout=dropout_prob)
        self.rq = _ResidualVQ(num_emb_list, e_dim, beta=beta,
                              kmeans_init=kmeans_init, kmeans_iters=kmeans_iters)
        self.decoder = _MLP(enc_dims[::-1], dropout=dropout_prob)

    def forward(self, x):
        z = self.encoder(x)
        z_q, rq_loss, indices = self.rq(z)
        out = self.decoder(z_q)
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, x):
        z = self.encoder(x)
        _, _, indices = self.rq(z)
        return indices

    def compute_loss(self, out, rq_loss, x):
        if self.loss_type == "mse":
            recon_loss = F.mse_loss(out, x)
        else:
            recon_loss = F.l1_loss(out, x)
        total = recon_loss + self.quant_loss_weight * rq_loss
        return total, recon_loss


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class _Trainer:
    def __init__(self, model, lr, weight_decay, epochs, eval_step, patience,
                 device, ckpt_dir):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.epochs = epochs
        self.eval_step = eval_step
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay

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
        return path

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
            for batch, _ in loader:
                batch = batch.to(self.device)
                out, rq_loss, _ = self.model(batch)
                loss, _ = self.model.compute_loss(out, rq_loss, batch)
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
# Helpers
# ---------------------------------------------------------------------------

def _default_layers(in_dim: int, e_dim: int) -> list:
    candidates = [2048, 1024, 512, 256, 128, 64]
    return [l for l in candidates if e_dim < l < in_dim]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_rqvae(
    embeddings: np.ndarray,
    ckpt_dir: str,
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
    verbose: bool = True,
) -> dict:
    """Train TIGER-style RQ-VAE and return index dict.

    Returns:
        {item_id (int): ["<a_X>", "<b_Y>", ...]}
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    n_items, in_dim = embeddings.shape
    os.makedirs(ckpt_dir, exist_ok=True)

    layers = _default_layers(in_dim, e_dim)
    if verbose:
        print(f"  TIGER RQ-VAE: {n_items} items, {in_dim}d -> {layers} -> {e_dim}d")

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

    dataset = _EmbDataset(emb_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    logging.basicConfig(level=logging.INFO)
    trainer = _Trainer(model, lr=lr, weight_decay=1e-4, epochs=epochs,
                       eval_step=eval_step, patience=patience,
                       device=device, ckpt_dir=ckpt_dir)
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

    # Generate indices (pure nearest-neighbor, no Sinkhorn)
    all_emb = torch.tensor(embeddings, dtype=torch.float32)
    all_indices = []
    with torch.no_grad():
        for start in range(0, n_items, batch_size):
            batch = all_emb[start:start + batch_size].to(dev)
            all_indices.append(model.get_indices(batch).cpu())
    all_indices = torch.cat(all_indices, dim=0)

    # Optional: ZCR for zero collision
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

    # Return tuple: (index_dict, codes_after, zcr_stats) so caller can write
    # provenance sidecars via zcr_provenance.write_zcr_sidecars().
    return index_dict, codes_np, zcr_stats
