import numpy as np


def recall_at_k(actual_list, pred_list, k=10):
    """
    Recall@K 메트릭 계산

    공식:
        Recall@K = (1/|U|) * Σ_u ( |I_u ∩ R_u(K)| / min(K, |I_u|) )

    Args:
        actual_list: 각 유저별 정답 아이템 리스트 (I_u)
        pred_list: 각 유저별 모델이 예측한 추천 리스트 (전체 순위)
        k: 평가할 추천 개수 (K)

    Returns:
        float: 전체 유저에 대한 평균 Recall@K
    """
    recalls = []

    for actual, pred in zip(actual_list, pred_list):
        if len(actual) == 0:
            continue

        actual_set = set(actual)
        pred_k = pred[:k]  # 상위 K개만 슬라이싱

        # 1. 분자: 추천한 K개 중 정답 개수
        hits = len(actual_set.intersection(pred_k))

        # 2. 분모: min(K, 유저가 실제로 본 아이템 수)
        denom = min(k, len(actual))

        # 3. 유저별 Recall 계산
        recalls.append(hits / denom)

    # 4. 전체 유저에 대한 평균 (1/|U|)
    return float(np.mean(recalls)) if recalls else 0.0
