import numpy as np
import os
from scipy import sparse

class BaseSlimModel(object):
    """
    기본 SLIM 모델 클래스.

    Methods:
        fit(X): 입력 데이터 행렬 X를 학습하고 계수를 초기화합니다.
        predict(X): 학습된 계수를 바탕으로 점수를 예측합니다.
        recommend(X, top): 상위 N개의 추천 아이템을 반환합니다.

    Args:
        X (scipy.sparse matrix): 사용자-아이템 상호작용 행렬.
        top (int, optional): 추천할 아이템 수. 기본값은 20.
    """
    def fit(self, X):
        self.coef = np.identity(X.shape[1])

    def predict(self, X):
        return X.dot(self.coef)

    def recommend(self, X, top=20):
        scores = self.predict(X)
        top_items = np.argsort(scores, axis=1)[:, -top:]
        return top_items

class AdmmSlim(BaseSlimModel):
    """
    ADMMSLIM 모델 클래스.

    ADMM 최적화 방법을 사용하여 SLIM(스파스 선형 항등 모델)을 학습합니다.

    Args:
        lambda_1 (float, optional): L1 정규화 항의 가중치. 기본값은 1.
        lambda_2 (float, optional): L2 정규화 항의 가중치. 기본값은 500.
        rho (float, optional): ADMM의 페널티 매개변수. 기본값은 10000.
        positive (bool, optional): 계수를 양수로 제한할지 여부. 기본값은 True.
        n_iter (int, optional): 최대 반복 횟수. 기본값은 50.
        eps_rel (float, optional): 상대 허용 오차. 기본값은 1e-4.
        eps_abs (float, optional): 절대 허용 오차. 기본값은 1e-3.
        verbose (bool, optional): 학습 중 로그를 출력할지 여부. 기본값은 False.

    Methods:
        fit(X): 사용자-아이템 상호작용 행렬을 사용해 모델을 학습합니다.
        soft_thresholding(B, Gamma): 소프트 임계값 조정을 수행합니다.
        is_converged(B, C, C_old, Gamma): 학습이 수렴했는지 판단합니다.
    """

    def __init__(self, lambda_1=1, lambda_2=500, rho=10000,
                 positive=True, n_iter=50, eps_rel=1e-4, eps_abs=1e-3,
                 verbose=False):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.rho = rho
        self.positive = positive
        self.n_iter = n_iter
        self.eps_rel = eps_rel
        self.eps_abs = eps_abs
        self.verbose = verbose

    def soft_thresholding(self, B, Gamma):
        if self.lambda_1 == 0:
            if self.positive:
                return np.abs(B)
            else:
                return B
        else:
            x = B + Gamma / self.rho
            threshold = self.lambda_1 / self.rho
            if self.positive:
                return np.where(threshold < x, x - threshold, 0)
            else:
                return np.where(threshold < x, x - threshold,
                                np.where(x < - threshold, x + threshold, 0))

    def is_converged(self, B, C, C_old, Gamma):
        B_norm = np.linalg.norm(B)
        C_norm = np.linalg.norm(C)
        Gamma_norm = np.linalg.norm(Gamma)

        eps_primal = self.eps_abs * B.shape[0] - self.eps_rel * np.max([B_norm, C_norm])
        eps_dual = self.eps_abs * B.shape[0] - self.eps_rel * Gamma_norm

        R_primal_norm = np.linalg.norm(B - C)
        R_dual_norm = np.linalg.norm(C  - C_old) * self.rho

        converged = R_primal_norm < eps_primal and R_dual_norm < eps_dual
        return converged

    def _evaluate(self, train_X, valid_X, k: int = 10):
        num_users = train_X.shape[0]
        recalls = []
        ndcgs = []

        for user_id in range(num_users):
            train_vec = train_X[user_id]
            valid_items = valid_X[user_id].indices
            if valid_items.size == 0:
                continue

            scores = self.predict(train_vec).ravel()
            seen = train_vec.indices
            scores[seen] = -np.inf

            k_eff = min(k, scores.shape[0])
            top_items = np.argpartition(scores, -k_eff)[-k_eff:]
            top_items = top_items[np.argsort(-scores[top_items])]

            valid_set = set(valid_items.tolist())
            hits = 0
            dcg = 0.0
            for rank, item in enumerate(top_items):
                if item in valid_set:
                    hits += 1
                    dcg += 1.0 / np.log2(rank + 2)

            recall = hits / float(valid_items.size)
            ideal = min(valid_items.size, k_eff)
            idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal + 2))))
            ndcg = 0.0 if idcg == 0.0 else (dcg / idcg)

            recalls.append(recall)
            ndcgs.append(ndcg)

        recall_mean = float(np.mean(recalls)) if recalls else 0.0
        ndcg_mean = float(np.mean(ndcgs)) if ndcgs else 0.0
        return recall_mean, ndcg_mean

    def fit(self, X, valid_X=None, eval_every: int = 10, eval_k: int = 10):
        XtX = X.T.dot(X)
        if sparse.issparse(XtX):
            XtX = XtX.todense().A

        if self.verbose:
            print(' --- init')
        identity_mat = np.identity(XtX.shape[0])
        diags = identity_mat * (self.lambda_2 + self.rho)
        P = np.linalg.inv(XtX + diags).astype(np.float32)
        B_aux = P.dot(XtX)

        Gamma = np.zeros_like(XtX, dtype=np.float32)
        C = np.zeros_like(XtX, dtype=np.float32)

        self.primal_residuals = []
        self.dual_residuals = []

        for iter in range(self.n_iter):
            if self.verbose:
                print(f'Iteration {iter+1}')
            C_old = C.copy()
            B_tilde = B_aux + P.dot(self.rho * C - Gamma)
            gamma = np.diag(B_tilde) / (np.diag(P) + 1e-8)
            B = B_tilde - P * gamma
            C = self.soft_thresholding(B, Gamma)
            Gamma = Gamma + self.rho * (B - C)

            R_primal_norm = np.linalg.norm(B - C)
            R_dual_norm = np.linalg.norm(C - C_old) * self.rho
            self.primal_residuals.append(R_primal_norm)
            self.dual_residuals.append(R_dual_norm)

            if self.verbose:
                print(f'     Primal Residual Norm: {R_primal_norm:.6f}')
                print(f'     Dual Residual Norm: {R_dual_norm:.6f}')

            if self.is_converged(B, C, C_old, Gamma):
                if self.verbose:
                    print(f' --- Converged at iteration {iter+1}. Stopped iteration.')
                break

            if valid_X is not None and eval_every and ((iter + 1) % int(eval_every) == 0):
                self.coef = C
                recall, ndcg = self._evaluate(X, valid_X, k=int(eval_k))
                if self.verbose:
                    print(' --- Validation')
                    print(f'     Validation recall@{int(eval_k)}: {recall:.6f}')
                    print(f'     Validation ndcg@{int(eval_k)}: {ndcg:.6f}')

        self.coef = C

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            coef=self.coef,
            lambda_1=np.array(self.lambda_1, dtype=np.float32),
            lambda_2=np.array(self.lambda_2, dtype=np.float32),
            rho=np.array(self.rho, dtype=np.float32),
            positive=np.array(int(self.positive), dtype=np.int32),
            n_iter=np.array(self.n_iter, dtype=np.int32),
            eps_rel=np.array(self.eps_rel, dtype=np.float32),
            eps_abs=np.array(self.eps_abs, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str) -> "AdmmSlim":
        data = np.load(path, allow_pickle=False)
        model = cls(
            lambda_1=float(data["lambda_1"]),
            lambda_2=float(data["lambda_2"]),
            rho=float(data["rho"]),
            positive=bool(int(data["positive"])),
            n_iter=int(data["n_iter"]),
            eps_rel=float(data["eps_rel"]),
            eps_abs=float(data["eps_abs"]),
            verbose=False,
        )
        model.coef = data["coef"]
        return model