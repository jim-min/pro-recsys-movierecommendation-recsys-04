import numpy as np


def recommend_topk(
    score_mat,
    train_mat,
    topk,
    mask_train=True,
):
    """
    score_mat: np.ndarray (num_users x num_items)
    train_mat: csr_matrix
    """

    scores = score_mat.copy()

    if mask_train:
        for u in range(train_mat.shape[0]):
            seen = train_mat.indices[
                train_mat.indptr[u] : train_mat.indptr[u + 1]
            ]
            scores[u, seen] = -1e9

    idx = np.argpartition(scores, -topk, axis=1)[:, -topk:]
    part = scores[np.arange(scores.shape[0])[:, None], idx]
    order = np.argsort(part, axis=1)[:, ::-1]

    return idx[np.arange(idx.shape[0])[:, None], order]
