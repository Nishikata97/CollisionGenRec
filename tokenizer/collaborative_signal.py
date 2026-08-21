"""Collaborative signal extraction: PPMI + truncated SVD.

Extracts symmetric co-occurrence structure from user interaction sequences,
then compresses via SVD. The resulting embeddings encode which items tend to
appear near each other.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import svd_flip


class CollaborativeSignal:

    def __init__(self, method="ppmi_svd", window_size=3, d_cf=256, holdout=2):
        if method != "ppmi_svd":
            raise ValueError(f"Unsupported method: {method}")
        self.method = method
        self.window_size = window_size
        self.d_cf = d_cf
        self.holdout = holdout

    def fit(self, interactions: dict, n_items: int) -> np.ndarray:
        """Compute (n_items, d_cf) L2-normalized collaborative embeddings."""
        return self._ppmi_svd(interactions, n_items)

    def _ppmi_svd(self, interactions: dict, n_items: int) -> np.ndarray:
        cooccur = self._build_cooccurrence_matrix(interactions, n_items)
        print(f"  Co-occurrence matrix: {n_items}x{n_items}, nnz={cooccur.nnz:,}")

        ppmi = self._ppmi_transform(cooccur)
        print(f"  PPMI matrix: nnz={ppmi.nnz:,}")

        k = min(self.d_cf, n_items - 2, ppmi.shape[0] - 2)
        print(f"  Computing truncated SVD (k={k})...")
        # Fixed ARPACK start vector so the collaborative signal - and thus the
        # CF SID assignment - is reproducible run-to-run. Without a pinned v0,
        # svds draws a random Lanczos start each call (svd_flip fixes only the
        # sign, not the subspace), making the CF variant non-deterministic.
        v0 = np.random.RandomState(42).standard_normal(min(ppmi.shape))
        U, S, Vt = svds(ppmi.astype(np.float64), k=k, v0=v0)

        # Resolve sign ambiguity: each column's sign is determined by the
        # element with largest absolute value, making SVD output deterministic.
        U, Vt = svd_flip(U, Vt)

        # Sort by descending singular value
        idx = np.argsort(-S)
        U, S = U[:, idx], S[idx]

        E_cf = U * np.sqrt(S)[np.newaxis, :]

        # L2 normalize
        norms = np.maximum(np.linalg.norm(E_cf, axis=1, keepdims=True), 1e-8)
        E_cf = E_cf / norms

        print(f"  Collaborative embeddings: {E_cf.shape}")
        return E_cf.astype(np.float32)

    def _build_cooccurrence_matrix(self, interactions: dict,
                                   n_items: int) -> sparse.csr_matrix:
        """Build symmetric co-occurrence matrix with holdout."""
        rows, cols, data = [], [], []
        for seq in interactions.values():
            seq_int = [int(x) for x in seq]
            if self.holdout > 0:
                seq_int = seq_int[:-self.holdout]
            if len(seq_int) < 2:
                continue
            for i, item_a in enumerate(seq_int):
                for j in range(i + 1, min(len(seq_int), i + self.window_size + 1)):
                    item_b = seq_int[j]
                    if item_a != item_b:
                        rows.extend([item_a, item_b])
                        cols.extend([item_b, item_a])
                        data.extend([1.0, 1.0])

        return sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_items, n_items), dtype=np.float64,
        )

    @staticmethod
    def _ppmi_transform(cooccur: sparse.csr_matrix) -> sparse.csr_matrix:
        """Positive Pointwise Mutual Information transform."""
        total = cooccur.sum()
        if total == 0:
            return cooccur

        row_sums = np.maximum(np.array(cooccur.sum(axis=1)).flatten(), 1e-12)
        col_sums = np.maximum(np.array(cooccur.sum(axis=0)).flatten(), 1e-12)

        coo = cooccur.tocoo()
        pmi = np.log(coo.data * total / (row_sums[coo.row] * col_sums[coo.col]))
        pmi = np.maximum(pmi, 0)

        return sparse.csr_matrix(
            (pmi, (coo.row, coo.col)), shape=cooccur.shape,
        )
