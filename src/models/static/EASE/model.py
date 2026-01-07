import numpy as np


class EASE:
    """
    EASE (Embarrassingly Shallow AutoEncoder)
    """
    def __init__(self, lambda_reg=300.0):
        self.lambda_reg = lambda_reg
        self.B = None

    def fit(self, X):
        """
        X: csr_matrix (num_users x num_items)
        """
        # 🔥 핵심 수정: sparse → dense
        G = (X.T @ X).toarray()

        diag_idx = np.arange(G.shape[0])
        G[diag_idx, diag_idx] += self.lambda_reg

        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        B[diag_idx, diag_idx] = 0.0

        self.B = B

    def predict(self, X):
        """
        X: csr_matrix
        return: dense score matrix
        """
        return X @ self.B
