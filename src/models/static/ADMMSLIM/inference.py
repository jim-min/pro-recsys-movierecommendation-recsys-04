import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import hydra
from omegaconf import DictConfig

from preprocessing import load_and_preprocess_data
from model import AdmmSlim

def generate_recommendations(model, user_item_matrix, num_users, K=10, future_delete=False, years=None, user_encoder=None, item_encoder=None):
    user_recommendations = {}
    
    # [최적화 1] 반복문 밖에서 '아이템 연도' 배열을 미리 만듭니다.
    # 아이템 ID를 인덱스로 사용하여 연도에 바로 접근(O(1))할 수 있게 합니다.
    item_years_array = None
    user_max_years = None
    
    if future_delete:
        # 유저별 마지막 시청 연도 계산
        users_year = pd.read_csv("/data/ephemeral/home/data/train/train_ratings_years.csv")
        # 인코딩이 필요하다면 적용 (이미 되어있다면 생략 가능)
        if user_encoder:
            users_year["user_id"] = user_encoder.transform(users_year['user'])
        else:
            users_year["user_id"] = users_year['user']
            
        user_max_years = users_year.groupby("user_id")["time"].max().to_dict()
        
        # 아이템 연도 배열 생성 (크기: 전체 아이템 수 + 1)
        # 기본값을 아주 큰 수(9999)로 설정하여, 연도 정보가 없는 아이템이 실수로 필터링되거나 추천되는 것을 방지
        num_items = user_item_matrix.shape[1]
        item_years_array = np.full(num_items, 9999) 
        
        # years 데이터프레임의 아이템 ID 인코딩
        if item_encoder:
             # years의 item 컬럼을 인코딩된 정수로 변환
            # (주의: years 데이터프레임 원본이 변경되지 않도록 copy 사용 권장)
            temp_items = item_encoder.transform(years['item'])
            temp_years = years['year'].values
        else:
            temp_items = years['item'].values
            temp_years = years['year'].values
            
        # 배열에 연도 채워넣기
        # 예: item_years_array[5] = 2022 (5번 아이템은 2022년 개봉)
        item_years_array[temp_items] = temp_years

    debug_count = 0
    for user_id in tqdm(range(num_users)):
        user_vector = user_item_matrix[user_id]
        scores = model.predict(user_vector)
        scores = scores.ravel()
        user_interacted_items = user_vector.indices
        scores[user_interacted_items] = -np.inf

        if future_delete:
            last_year = user_max_years.get(user_id)
            
            if last_year is not None:
                cutoff_year = last_year

                # [디버깅] 실제 데이터 확인
                if debug_count < 5: # 처음 5명만 확인
                    # 1. 인코딩 매핑 확인
                    # 현재 유저가 실제로 본 아이템 중 하나를 골라서 연도가 제대로 매핑됐나 확인
                    if len(user_interacted_items) > 0:
                        sample_item_idx = user_interacted_items[0]
                        print(f"\n[User {user_id}] Last Year: {last_year}")
                        print(f" - Sample Interacted Item Index: {sample_item_idx}")
                        print(f" - Mapped Year in Array: {item_years_array[sample_item_idx]}") 
                        # 만약 여기서 9999나 0이 나온다면 매핑 잘못된 것임!

                    # 2. 필터링 강도 확인
                    future_mask = item_years_array > cutoff_year
                    filtered_count = np.sum(future_mask)
                    total_items = len(scores)
                    print(f" - Filtered Items: {filtered_count} / {total_items} ({(filtered_count/total_items)*100:.1f}%)")
                    
                    debug_count += 1
                
                # Numpy Boolean Indexing: 한 번에 비교하여 마스킹 (속도 매우 빠름)
                # item_years_array의 모든 값과 cutoff_year를 한 번에 비교
                future_mask = item_years_array > cutoff_year
                
                # 미래 아이템 점수를 -inf로 변경
                scores[future_mask] = -np.inf

        top_items = np.argpartition(scores, -K)[-K:]
        top_items = top_items[np.argsort(-scores[top_items])]
        user_recommendations[user_id] = top_items

    return user_recommendations


@hydra.main(version_base=None, config_path="../../config/SLIM", config_name="config")
def main(cfg: DictConfig):
    data_path = cfg.dataset.data_path
    years_path = cfg.dataset.years_path

    train_df, years_df, train_matrix, _, user_encoder, item_encoder = load_and_preprocess_data(data_path, years_path=years_path)

    model_path = cfg.get("output", {}).get("model_path", None)
    if not model_path:
        raise ValueError("cfg.output.model_path is required for inference")

    model = AdmmSlim.load(model_path)

    topk = int(cfg.get("inference", {}).get("topk", 10))
    user_recommendations = generate_recommendations(
        model=model,
        user_item_matrix=train_matrix,
        num_users=train_df["user_id"].nunique(),
        K=topk,
        future_delete=cfg.inference.future_delete,
        years=years_df,
        user_encoder=user_encoder,
        item_encoder=item_encoder
    )

    recommendations = []
    for user_id, item_ids in tqdm(user_recommendations.items(), desc="Saving Recommendations"):
        user_original_id = user_encoder.inverse_transform([user_id])[0]
        for item_id in item_ids:
            item_original_id = item_encoder.inverse_transform([item_id])[0]
            recommendations.append({"user": user_original_id, "item": item_original_id})

    recommendations_df = pd.DataFrame(recommendations)
    recommendations_path = cfg.get("output", {}).get("recommendations_path", "recommendations.csv")
    os.makedirs(os.path.dirname(recommendations_path) or ".", exist_ok=True)
    recommendations_df.to_csv(recommendations_path, index=False)


if __name__ == "__main__":
    main()