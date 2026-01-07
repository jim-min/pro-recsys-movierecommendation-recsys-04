import numpy as np
from scipy.sparse import csr_matrix


class SLIM:
    """
    SLIM (Sparse Linear Methods) item-item model
    Stores W (item-item similarity/weight matrix), diagonal is forced to 0.
    """
    def __init__(self):
        self.W = None  # csr_matrix (num_items x num_items)

    def set_W(self, W: csr_matrix):
        self.W = W

    def predict_scores(self, X_user_item):
        """
        X_user_item: csr_matrix (num_users x num_items)
        return: csr_matrix or dense-like scores (num_users x num_items)
        """
        if self.W is None:
            raise RuntimeError("Model is not trained. W is None.")
        return X_user_item @ self.W
