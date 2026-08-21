"""Embedding fusion: concat + PCA.

Combines L2-normalized semantic embeddings with weighted collaborative
embeddings, then reduces dimensionality via PCA.
"""

import numpy as np
from sklearn.decomposition import PCA


def fuse_embeddings(E_sem: np.ndarray, E_cf: np.ndarray,
                    alpha: float, target_dim: int = None) -> np.ndarray:
    """Fuse semantic and collaborative embeddings via concat + PCA.

    Steps: L2-normalize E_sem → concat [E_sem_norm, alpha * E_cf] → PCA.

    Args:
        E_sem: Semantic embeddings (n_items, d_text)
        E_cf: Collaborative embeddings (n_items, d_cf), pre-normalized
        alpha: Weight for collaborative component
        target_dim: PCA output dimension (default: d_text)

    Returns:
        Fused embeddings (n_items, target_dim)
    """
    if target_dim is None:
        target_dim = E_sem.shape[1]

    sem_norms = np.maximum(np.linalg.norm(E_sem, axis=1, keepdims=True), 1e-8)
    E_sem_norm = E_sem / sem_norms

    E_combined = np.hstack([E_sem_norm, alpha * E_cf])
    print(f"  Combined shape before PCA: {E_combined.shape}")

    n_components = min(target_dim, E_combined.shape[0] - 1, E_combined.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    E_reduced = pca.fit_transform(E_combined)
    explained_var = sum(pca.explained_variance_ratio_)
    print(f"  PCA: {E_combined.shape[1]} -> {n_components} dims, "
          f"explained variance: {explained_var:.4f}")

    return E_reduced.astype(np.float32)


