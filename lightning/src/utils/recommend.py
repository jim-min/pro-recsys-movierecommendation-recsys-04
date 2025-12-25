import torch
import numpy as np


def recommend_topk(model, train_mat, k=10, device="cuda", batch_size=512, exclude_items=None):
    """
    Top-K 추천 생성 (배치 단위 처리)

    Args:
        model: 학습된 MultiVAE 모델
        train_mat: 학습 데이터 sparse matrix (CSR)
        k: 추천할 아이템 개수
        device: 연산 디바이스
        batch_size: 배치 크기
        exclude_items: Dict[user_idx, Set[item_idx]] - 추천에서 제외할 아이템 (future items 등)

    Returns:
        np.ndarray: (num_users, k) 각 유저별 top-k 추천 아이템 인덱스
    """
    model.eval()
    model.to(device)

    num_users = train_mat.shape[0]
    recommendations = []

    with torch.no_grad():
        for start_idx in range(0, num_users, batch_size):
            end_idx = min(start_idx + batch_size, num_users)

            # Batch 데이터 준비
            batch_mat = train_mat[start_idx:end_idx].toarray()
            batch_tensor = torch.FloatTensor(batch_mat).to(device)

            # 추론
            logits, _, _ = model(batch_tensor)
            scores = logits.cpu().numpy()

            # 이미 본 아이템 제거 (학습 데이터에 있는 아이템은 추천하지 않음)
            scores[batch_mat.nonzero()] = -np.inf

            # Future items 제거 (year filtering)
            if exclude_items is not None:
                for batch_idx, user_idx in enumerate(range(start_idx, end_idx)):
                    if user_idx in exclude_items:
                        future_items = list(exclude_items[user_idx])
                        if future_items:
                            scores[batch_idx, future_items] = -np.inf

            # Top-K 추출
            topk_indices = np.argsort(-scores, axis=1)[:, :k]
            recommendations.append(topk_indices)

    recommendations = np.vstack(recommendations)
    return recommendations
