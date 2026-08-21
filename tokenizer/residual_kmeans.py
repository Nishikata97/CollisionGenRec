"""Residual K-Means (RK-Means) for hierarchical vector quantization.

Each layer runs FAISS K-Means on the residual from previous layers,
producing an L-token SID per item.
"""

import torch
from torch import nn


class ResKmeans(nn.Module):

    def __init__(self, n_layers, codebook_size, dim):
        super().__init__()
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.dim = dim
        self.centroids = nn.ParameterList([
            nn.Parameter(torch.zeros(codebook_size, dim), requires_grad=False)
            for _ in range(n_layers)
        ])

    def train_kmeans(self, inputs, niter=20, verbose=True, seed=42):
        """Train RK-Means layer by layer: FAISS K-Means → residual → next."""
        import faiss
        x = inputs.clone().numpy().astype("float32")
        out = torch.zeros_like(inputs)

        for l in range(self.n_layers):
            kmeans = faiss.Kmeans(
                self.dim, self.codebook_size, niter=niter,
                verbose=verbose, spherical=False, seed=seed + l,
            )
            kmeans.train(x)
            _, I = kmeans.index.search(x, 1)
            I = I.reshape(-1)
            centroids_np = kmeans.centroids.copy()

            out += torch.tensor(centroids_np[I])
            if verbose:
                loss = ((inputs - out) ** 2).mean().item()
                print(f"  Layer {l}: mse={loss:.6f}")

            x = x - centroids_np[I]
            self.centroids[l] = nn.Parameter(
                torch.tensor(centroids_np), requires_grad=False,
            )

    def encode(self, x):
        """Encode embeddings → (n_items, n_layers) code tensor."""
        codes = []
        for l in range(self.n_layers):
            x_sq = x.pow(2).sum(dim=1, keepdim=True)
            c_sq = self.centroids[l].T.pow(2).sum(dim=0, keepdim=True)
            dists = torch.addmm(x_sq + c_sq, x, self.centroids[l].T, alpha=-2.0)
            code = dists.argmin(dim=-1)
            x = x - self.centroids[l][code]
            codes.append(code)
        return torch.stack(codes, dim=1)

    def decode(self, codes):
        """Decode codes → reconstructed embeddings."""
        out = torch.zeros(codes.shape[0], self.dim, dtype=torch.float32,
                          device=codes.device)
        for l in range(codes.shape[1]):
            out += self.centroids[l][codes[:, l]]
        return out
