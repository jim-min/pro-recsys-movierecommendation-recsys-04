import torch
import numpy as np

@torch.no_grad()
def recommend_topk(
    model,
    train_mat,
    topk,
    device,
    batch_size=512,
    mask_train=True,
):
    """
    model: trained model
    train_mat: user-item matrix used for training
    topk: number of items to return per user (10 or 100)
    mask_train: True → mask items seen in train_mat
    """

    model.eval()
    num_users = train_mat.shape[0]
    all_recs = []

    for start in range(0, num_users, batch_size):
        end = min(start + batch_size, num_users)

        x = torch.from_numpy(
            train_mat[start:end].toarray()
        ).float().to(device)

        logits, _, _ = model(x)
        scores = logits.cpu().numpy()

        if mask_train:
            for i, u_idx in enumerate(range(start, end)):
                seen_items = train_mat[u_idx].indices
                scores[i, seen_items] = -1e9

        topk_idx = np.argsort(scores, axis=1)[:, -topk:][:, ::-1]
        all_recs.append(topk_idx)

    return np.concatenate(all_recs, axis=0)
