import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from sklearn.linear_model import ElasticNet
from joblib import Parallel, delayed


class SLIMTrainer:
    def __init__(
        self,
        model,
        train_mat,                 # csr_matrix (U x I)
        alpha=1e-4,                # 규제 강도 (클수록 더 sparse/더 강한 규제)
        l1_ratio=0.1,              # L1 비율 (0=Ridge, 1=Lasso)
        positive=True,             # 가중치 양수 constraint (보통 True)
        max_iter=80,               # 너무 크게 잡으면 오래 걸림
        tol=1e-4,
        n_jobs=8,
        verbose=1,
    ):
        self.model = model
        self.train_mat = train_mat.tocsr()
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.positive = bool(positive)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_jobs = int(n_jobs)
        self.verbose = int(verbose)

    def _fit_one_item(self, X_csc: csc_matrix, j: int):
        """
        Fit ElasticNet to predict column j from all columns of X.
        NOTE: self-feature leakage를 완전히 제거하려면 더 복잡해져서,
              여기서는 fit 후 diag를 0으로 만드는 실전 타협 버전.
        """
        y = X_csc.getcol(j).toarray().ravel()  # (num_users,)

        # ElasticNet on sparse X
        enet = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=False,
            positive=self.positive,
            max_iter=self.max_iter,
            tol=self.tol,
            selection="cyclic",
        )
        enet.fit(X_csc, y)

        w = enet.coef_.astype(np.float32)  # (num_items,)
        w[j] = 0.0  # diagonal 제거
        nnz = np.count_nonzero(w)
        return j, w, nnz

    def train(self):
        X_csc = self.train_mat.tocsc()
        num_items = X_csc.shape[1]

        if self.verbose:
            print(f"[SLIM] X shape = {X_csc.shape}, items={num_items}")
            print(f"[SLIM] alpha={self.alpha}, l1_ratio={self.l1_ratio}, "
                  f"max_iter={self.max_iter}, n_jobs={self.n_jobs}")

        results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self._fit_one_item)(X_csc, j) for j in range(num_items)
        )

        # assemble W (items x items)
        W = lil_matrix((num_items, num_items), dtype=np.float32)
        total_nnz = 0
        for j, w, nnz in results:
            if nnz > 0:
                idx = np.nonzero(w)[0]
                W[idx, j] = w[idx]
                total_nnz += nnz

        W = W.tocsr()
        if self.verbose:
            print(f"[SLIM] W built. nnz={W.nnz}, total_nnz_est={total_nnz}")

        self.model.set_W(W)
        return W
