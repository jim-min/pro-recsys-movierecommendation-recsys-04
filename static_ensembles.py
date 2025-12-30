import numpy as np
import pandas as pd

# 1. 데이터 로드
# 모델 A (Static - MVAE)
df_mvae = pd.read_csv("/data/ephemeral/home/jimin/pro-recsys-movierecommendation-recsys-04/mvae1.csv") 
# 모델 B (Static - EASE)
df_ease = pd.read_csv("/data/ephemeral/home/jimin/pro-recsys-movierecommendation-recsys-04/ease.csv") 

# 2. 각 모델의 추천 결과를 유저별 리스트로 변환
# (user_id를 키로 하고, item_list를 값으로 갖는 딕셔너리 생성)
def get_user_item_dict(df):
    # 유저별로 그룹화하여 아이템을 리스트로 만듦
    return df.groupby('user')['item'].apply(list).to_dict()

user_items_mvae = get_user_item_dict(df_mvae)
user_items_ease = get_user_item_dict(df_ease)

# 모든 유저 리스트 (두 모델 중 하나라도 있는 유저)
all_users = user_items_ease.keys()

result_user = []
result_item = []

k = 10  # RRF 파라미터 (보통 60 사용)

# 3. 유저별 RRF 계산
for user in all_users:
    item_scores = {}
    
    # (1) MVAE 모델 점수 계산
    if user in user_items_mvae:
        # 리스트의 인덱스(0부터 시작)가 곧 순위 정보임
        for rank, item in enumerate(user_items_mvae[user]):
            # 1 / (k + 등수) -> 등수는 1부터 시작하도록 rank+1
            score = 0.86 / (k + rank + 1)
            if item in item_scores:
                item_scores[item] += score
            else:
                item_scores[item] = score
    
    # (2) EASE 모델 점수 계산
    if user in user_items_ease:
        for rank, item in enumerate(user_items_ease[user]):
            score = 1 / (k + rank + 1)
            if item in item_scores:
                item_scores[item] += score
            else:
                item_scores[item] = score
    
    # (3) 점수순 정렬 후 Top 10 추출
    # 점수가 높은 순서대로 정렬
    sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 상위 10개 아이템만 가져오기
    top_10_items = [item for item, score in sorted_items[:10]]
    
    # 결과 저장용 리스트에 추가
    result_user.extend([user] * len(top_10_items))
    result_item.extend(top_10_items)

# 4. 결과 저장
answers = pd.DataFrame({
    "user": result_user,
    "item": result_item
})

answers.to_csv("/data/ephemeral/home/jimin/pro-recsys-movierecommendation-recsys-04/rrf_ensemble.csv", index=False)
print("RRF 앙상블 완료!")