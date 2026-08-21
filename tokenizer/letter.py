"""LETTER-style RQ-VAE tokenizer (Wang et al., CIKM 2024).

Faithful reproduction of LETTER's three regularizations:
  1. L_Sem: standard RQ-VAE (reconstruction + commitment)
  2. L_CF:  collaborative contrastive loss (pulls quantized embeddings
           toward CF embeddings via InfoNCE)
  3. L_Div: diversity regularization (constrained K-means clustering +
           contrastive loss on codebook embeddings)

Plus Sinkhorn balanced assignment on the last VQ level.

Uses PPMI+SVD for CF embeddings instead of SASRec/LightGCN for
controlled comparison with other tokenizers.

Reference: "Learnable Item Tokenization for Generative Recommendation"
(CIKM 2024, arXiv 2405.07314)
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
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset

from rqvae import _default_layers, _MLP


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _EmbIdxDataset(Dataset):
    def __init__(self, emb_path):
        self.data = np.load(emb_path).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), idx


# ---------------------------------------------------------------------------
# Sinkhorn
# ---------------------------------------------------------------------------

@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, iters):
    Q = torch.exp(-distances / epsilon)
    B, K = Q.shape
    Q /= Q.sum()
    for _ in range(iters):
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= B
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= K
    Q *= B
    return Q


# ---------------------------------------------------------------------------
# VQ with Sinkhorn + diversity loss
# ---------------------------------------------------------------------------

class _LetterVQ(nn.Module):
    def __init__(self, n_e, e_dim, mu=0.25, beta=0.1,
                 kmeans_init=True, kmeans_iters=10,
                 sk_epsilon=0.0, sk_iters=50,
                 n_clusters=10):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.mu = mu          # commitment weight (LETTER calls it mu)
        self.beta = beta      # diversity weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.n_clusters = n_clusters
        self.embedding = nn.Embedding(n_e, e_dim)
        if not kmeans_init:
            self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)
            self.initted = True
        else:
            self.embedding.weight.data.zero_()
            self.initted = False
        self.cluster_labels = None  # set by trainer

    def _kmeans_init(self, data):
        from k_means_constrained import KMeansConstrained
        x = data.cpu().detach().numpy()
        n_clusters = self.n_e
        size_min = min(len(x) // (n_clusters * 2), 50)
        size_max = max(size_min * 4, -(-len(x) // n_clusters))
        clf = KMeansConstrained(
            n_clusters=n_clusters, size_min=size_min, size_max=size_max,
            max_iter=10, n_init=10, n_jobs=1, verbose=False)
        clf.fit(x)
        self.embedding.weight.data.copy_(
            torch.from_numpy(clf.cluster_centers_).to(data.device))
        self.initted = True

    def _compute_distances(self, x):
        return (x ** 2).sum(1, keepdim=True) \
            + (self.embedding.weight ** 2).sum(1).unsqueeze(0) \
            - 2 * x @ self.embedding.weight.t()

    @staticmethod
    def _center_distance(d):
        mid = (d.max() + d.min()) / 2
        amp = d.max() - mid + 1e-5
        return (d - mid) / amp

    def _diversity_loss(self, x_q, indices):
        """LETTER's L_Div: contrastive loss within constrained K-means clusters."""
        if self.cluster_labels is None or self.beta <= 0:
            return torch.tensor(0.0, device=x_q.device)

        labels = torch.tensor(self.cluster_labels, dtype=torch.long,
                              device=x_q.device)
        B = len(indices)
        cluster_of = labels[indices]  # (B,)

        # Same-cluster mask, excluding self
        same_cluster = labels.unsqueeze(0) == cluster_of.unsqueeze(1)  # (B, n_e)
        same_cluster[torch.arange(B, device=x_q.device), indices] = False

        # Sample positive: random same-cluster entry
        rand_scores = torch.where(
            same_cluster,
            torch.rand(same_cluster.shape, device=x_q.device),
            torch.full(same_cluster.shape, -1.0, device=x_q.device),
        )
        y_true = rand_scores.argmax(dim=1)

        # Similarity with self-masking
        sim = x_q @ self.embedding.weight.t()
        sim_self = torch.zeros_like(sim)
        sim_self.scatter_(1, indices.unsqueeze(1), 1e12)
        loss = F.cross_entropy(sim - sim_self, y_true)
        return loss

    def forward(self, x, use_sk=True):
        flat = x.view(-1, self.e_dim)
        if not self.initted and self.training:
            self._kmeans_init(flat)

        d = self._compute_distances(flat)

        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d_centered = self._center_distance(d).double()
            Q = sinkhorn_algorithm(d_centered, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                indices = torch.argmin(d, dim=-1)
            else:
                indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # Diversity loss
        div_loss = self._diversity_loss(x_q, indices)

        # Standard VQ losses
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.mu * commitment_loss + self.beta * div_loss

        x_q = x + (x_q - x).detach()  # STE
        return x_q, loss, indices.view(x.shape[:-1])


class _LetterRQ(nn.Module):
    def __init__(self, n_e_list, e_dim, mu=0.25, beta=0.1,
                 kmeans_init=True, kmeans_iters=10,
                 sk_epsilons=None, sk_iters=50, n_clusters=10):
        super().__init__()
        if sk_epsilons is None:
            sk_epsilons = [0.0] * (len(n_e_list) - 1) + [0.003]
        self.vq_layers = nn.ModuleList([
            _LetterVQ(n_e, e_dim, mu=mu, beta=beta,
                      kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                      sk_epsilon=sk_eps, sk_iters=sk_iters,
                      n_clusters=n_clusters)
            for n_e, sk_eps in zip(n_e_list, sk_epsilons)
        ])

    def forward(self, x, use_sk=True):
        all_losses, all_indices = [], []
        residual = x
        x_q = 0
        for vq in self.vq_layers:
            x_res, loss, indices = vq(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(indices)
        return x_q, torch.stack(all_losses).mean(), torch.stack(all_indices, dim=-1)


class LetterRQVAE(nn.Module):
    def __init__(self, in_dim, num_emb_list, e_dim, layers,
                 dropout_prob=0.0, loss_type="mse", quant_loss_weight=1.0,
                 mu=0.25, beta=0.1, alpha=0.1,
                 kmeans_init=True, kmeans_iters=10,
                 sk_epsilons=None, sk_iters=50, n_clusters=10):
        super().__init__()
        self.quant_loss_weight = quant_loss_weight
        self.loss_type = loss_type
        self.alpha = alpha
        enc_dims = [in_dim] + layers + [e_dim]
        self.encoder = _MLP(enc_dims, dropout=dropout_prob)
        self.rq = _LetterRQ(num_emb_list, e_dim, mu=mu, beta=beta,
                            kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                            sk_epsilons=sk_epsilons, sk_iters=sk_iters,
                            n_clusters=n_clusters)
        self.decoder = _MLP(enc_dims[::-1], dropout=dropout_prob)

    def forward(self, x, use_sk=True):
        z = self.encoder(x)
        z_q, rq_loss, indices = self.rq(z, use_sk=use_sk)
        out = self.decoder(z_q)
        return out, rq_loss, indices, z_q

    @torch.no_grad()
    def get_indices(self, x, use_sk=False):
        z = self.encoder(x)
        _, _, indices = self.rq(z, use_sk=use_sk)
        return indices

    def compute_loss(self, out, rq_loss, x):
        if self.loss_type == "mse":
            recon_loss = F.mse_loss(out, x)
        else:
            recon_loss = F.l1_loss(out, x)
        total = recon_loss + self.quant_loss_weight * rq_loss
        return total, recon_loss

    def cf_loss(self, z_q, cf_emb_batch):
        """L_CF: InfoNCE between quantized embeddings and CF embeddings."""
        B = z_q.size(0)
        labels = torch.arange(B, dtype=torch.long, device=z_q.device)
        sim = z_q @ cf_emb_batch.t()
        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# PPMI construction
# ---------------------------------------------------------------------------

def build_cf_embeddings(interactions, n_items, d_cf=256,
                        window_size=3, holdout=2):
    """Build CF embeddings via PPMI + SVD."""
    from collaborative_signal import CollaborativeSignal
    cs = CollaborativeSignal(method="ppmi_svd", window_size=window_size,
                             d_cf=d_cf, holdout=holdout)
    return cs.fit(interactions, n_items)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class _LetterTrainer:
    def __init__(self, model, lr, weight_decay, epochs, eval_step, patience,
                 device, ckpt_dir, cf_embeddings, n_clusters=10):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.epochs = epochs
        self.eval_step = eval_step
        self.patience = patience
        self.cf_embeddings = cf_embeddings  # (n_items, d_cf) numpy
        self.n_clusters = n_clusters

        ts = time.strftime("%b-%d-%Y_%H-%M-%S")
        self.ckpt_dir = os.path.join(ckpt_dir, ts)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        self.best_collision_rate = float("inf")
        self.logger = logging.getLogger(__name__)

    def _recompute_cluster_labels(self):
        """Recompute constrained K-means labels for each VQ layer's codebook."""
        from k_means_constrained import KMeansConstrained
        for vq in self.model.rq.vq_layers:
            emb = vq.embedding.weight.cpu().detach().numpy()
            nc = self.n_clusters
            size_min = min(len(emb) // (nc * 2), 10)
            size_max = max(nc * 6, -(-len(emb) // nc))
            clf = KMeansConstrained(
                n_clusters=nc, size_min=size_min, size_max=size_max,
                max_iter=10, n_init=10, n_jobs=1, verbose=False)
            clf.fit(emb)
            vq.cluster_labels = clf.labels_.tolist()

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
            indices = self.model.get_indices(batch, use_sk=False)
            all_codes.extend(indices.cpu().tolist())
        from collections import Counter
        code_strs = [str(c) for c in all_codes]
        n = len(code_strs)
        if n == 0:
            return 0.0
        group_sizes = Counter(code_strs)
        items_in_collision = sum(s for s in group_sizes.values() if s > 1)
        return items_in_collision / n

    def _vq_init_full(self, loader):
        """Initialize all VQ codebooks on full dataset in residual order
        (LETTER official convention: vq_initialization before training)."""
        self.model.eval()
        all_data = []
        for batch, _ in loader:
            all_data.append(batch)
        all_data = torch.cat(all_data, dim=0).to(self.device)
        with torch.no_grad():
            z = self.model.encoder(all_data)
            residual = z
            for vq in self.model.rq.vq_layers:
                vq._kmeans_init(residual)
                # Compute quantized and residual for next layer
                d = vq._compute_distances(residual)
                indices = torch.argmin(d, dim=-1)
                x_q = vq.embedding(indices)
                residual = residual - x_q
        self.logger.info("VQ codebooks initialized on full dataset")

    def fit(self, loader):
        # Full-dataset VQ initialization (LETTER convention)
        self._vq_init_full(loader)

        cf_emb_tensor = torch.tensor(self.cf_embeddings, dtype=torch.float32,
                                     device=self.device)
        best_loss = float("inf")
        for epoch in range(self.epochs):
            self._recompute_cluster_labels()

            self.model.train()
            epoch_loss = 0
            for batch, batch_idx in loader:
                batch = batch.to(self.device)
                out, rq_loss, indices, z_q = self.model(batch, use_sk=True)
                loss_total, _ = self.model.compute_loss(out, rq_loss, batch)

                # L_CF: collaborative contrastive loss
                if self.model.alpha > 0:
                    cf_batch = cf_emb_tensor[batch_idx]
                    loss_cf = self.model.cf_loss(z_q, cf_batch)
                    loss_total = loss_total + self.model.alpha * loss_cf

                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()
                epoch_loss += loss_total.item()

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

def train_rqvae(
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
    # LETTER-specific
    alpha: float = 0.1,
    beta: float = 0.1,
    n_clusters: int = 10,
    sk_iters: int = 50,
    d_cf: int = 32,  # must match e_dim for CF dot-product loss
    window_size: int = 3,
    holdout: int = 2,
    verbose: bool = True,
) -> dict:
    """Train LETTER-style RQ-VAE and return index dict."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    n_items, in_dim = embeddings.shape
    os.makedirs(ckpt_dir, exist_ok=True)

    # LETTER official uses fixed layers [2048,1024,512,256,128,64]
    # regardless of in_dim (encoder can expand beyond input dim)
    layers = [2048, 1024, 512, 256, 128, 64]
    if verbose:
        print(f"  LETTER RQ-VAE: {n_items} items, {in_dim}d -> {layers} -> {e_dim}d")
        print(f"  alpha={alpha}, beta={beta}, n_clusters={n_clusters}")

    # Build CF embeddings via PPMI+SVD
    if interactions is not None and alpha > 0:
        if verbose:
            print(f"  Building CF embeddings (PPMI+SVD, d_cf={d_cf})...")
        cf_embeddings = build_cf_embeddings(
            interactions, n_items, d_cf=d_cf,
            window_size=window_size, holdout=holdout)
        if verbose:
            print(f"  CF embeddings: {cf_embeddings.shape}")
    else:
        cf_embeddings = np.zeros((n_items, d_cf), dtype=np.float32)

    emb_path = os.path.join(ckpt_dir, "rqvae_input_emb.npy")
    np.save(emb_path, embeddings)

    sk_epsilons = [0.0] * (n_layers - 1) + [0.003]
    model = LetterRQVAE(
        in_dim=in_dim,
        num_emb_list=[codebook_size] * n_layers,
        e_dim=e_dim,
        layers=layers,
        mu=0.25, beta=beta, alpha=alpha,
        kmeans_init=True, kmeans_iters=100,
        sk_epsilons=sk_epsilons, sk_iters=sk_iters,
        n_clusters=n_clusters,
    )

    dataset = _EmbIdxDataset(emb_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    logging.basicConfig(level=logging.INFO)
    trainer = _LetterTrainer(
        model, lr=lr, weight_decay=1e-4, epochs=epochs,
        eval_step=eval_step, patience=patience,
        device=device, ckpt_dir=ckpt_dir,
        cf_embeddings=cf_embeddings, n_clusters=n_clusters,
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

    # Recompute cluster labels for Sinkhorn de-collision
    from k_means_constrained import KMeansConstrained
    for vq in model.rq.vq_layers:
        emb = vq.embedding.weight.cpu().detach().numpy()
        nc = n_clusters
        size_min = min(len(emb) // (nc * 2), 10)
        size_max = max(nc * 6, -(-len(emb) // nc))
        clf = KMeansConstrained(
            n_clusters=nc, size_min=size_min, size_max=size_max,
            max_iter=10, n_init=10, n_jobs=1, verbose=False)
        clf.fit(emb)
        vq.cluster_labels = clf.labels_.tolist()

    # Generate indices (no Sinkhorn at inference)
    all_emb = torch.tensor(embeddings, dtype=torch.float32)
    all_indices = []
    with torch.no_grad():
        for start in range(0, n_items, batch_size):
            batch = all_emb[start:start + batch_size].to(dev)
            all_indices.append(model.get_indices(batch, use_sk=False).cpu())
    all_indices = torch.cat(all_indices, dim=0)

    # Sinkhorn post-training de-collision (LETTER convention)
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    code_strs = [str(all_indices[i].tolist()) for i in range(n_items)]
    for iteration in range(20):
        groups = {}
        for i, c in enumerate(code_strs):
            groups.setdefault(c, []).append(i)
        collision_groups = [ids for ids in groups.values() if len(ids) > 1]
        if not collision_groups:
            break
        if verbose:
            print(f"  Sinkhorn iter {iteration}: "
                  f"{len(collision_groups)} collision groups")
        with torch.no_grad():
            for group in collision_groups:
                d = all_emb[group].to(dev)
                new_idx = model.get_indices(d, use_sk=True).cpu()
                for pos, item_id in enumerate(group):
                    all_indices[item_id] = new_idx[pos]
                    code_strs[item_id] = str(all_indices[item_id].tolist())

    n_unique_sk = len(set(code_strs))
    if verbose:
        print(f"  After Sinkhorn: {n_unique_sk}/{n_items} unique")

    # Optional ZCR
    if zcr:
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
