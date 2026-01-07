import numpy as np


class EASE:
    """
    EASE with normalized side-information similarity
    """

    def __init__(self, lambda_reg=300.0, meta_weights=None):
        self.lambda_reg = lambda_reg
        self.meta_weights = meta_weights or {}
        self.B = None

    def _normalize_similarity(self, S: np.ndarray):
        """
        Normalize similarity matrix by mean of non-zero entries
        """
        mask = S > 0
        if not np.any(mask):
            return S

        mean_val = S[mask].mean()
        if mean_val <= 0:
            return S

        return S / mean_val

    def fit(self, X, feat_mats=None):
        """
        X: user-item interaction matrix (csr_matrix)
        feat_mats: dict[str, csr_matrix]
        """
        print("🚀 Computing Interaction Gram Matrix (XᵀX)...")
        G = (X.T @ X).toarray()

        if feat_mats is not None:
            for name, mat in feat_mats.items():
                w = self.meta_weights.get(name, 0.0)
                if w <= 0:
                    continue

                print(f"➕ Adding {name} similarity (weight={w})")

                S = (mat @ mat.T).toarray()
                S = self._normalize_similarity(S)

                G += w * S

        print("🔧 Regularization & Inversion...")
        diag = np.arange(G.shape[0])
        G[diag, diag] += self.lambda_reg

        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        B[diag, diag] = 0.0

        self.B = B

    def predict(self, X):
        return X @ self.B
