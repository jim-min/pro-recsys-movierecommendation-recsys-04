import pandas as pd
import numpy as np
import json
from catboost import CatBoostRanker, Pool

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

# 1. 준비된 데이터 (이미 학습된 모델이 있다고 가정)
# movie_features: (6807, 영화피처수)

def load_movie_features(cfg):
    # 2. 개봉 연도 (years.tsv)
    df_years = pd.read_csv(to_absolute_path(cfg.data.year_path), sep='\t')
    df_years.columns = ['item', 'year']
    df_years['item'] = df_years['item'].astype(int)
    
    # 3. 감독 데이터 (directors.tsv)
    df_directors = pd.read_csv(to_absolute_path(cfg.data.directors_path), sep='\t')
    df_directors.columns = ['item', 'director']
    df_directors['item'] = df_directors['item'].astype(int)
    # 한 영화에 감독이 여러 명일 경우 하나로 합침 (중복 행 방지)
    df_directors = df_directors.groupby('item')['director'].apply(lambda x: ", ".join(x)).reset_index()

    # 4. 장르 데이터 (JSON)
    with open(to_absolute_path(cfg.data.genre_path), 'r') as f:
        genre_dict = json.load(f)
    df_genre = pd.DataFrame(list(genre_dict.items()), columns=['item', 'genre'])
    df_genre['item'] = df_genre['item'].astype(int)

    # 5. 모든 데이터 하나로 병합 (Left Join)
    # 모든 영화 정보를 가지고 있는 titles를 기준으로 나머지 정보들을 붙입니다.
    movie_features = df_years.merge(df_directors, on='item', how='left') \
                              .merge(df_genre, on='item', how='left')

    # 결측치 처리 (정보가 없는 경우 'unknown' 등으로 채워야 CatBoost 에러를 방지함)
    movie_features['year'] = movie_features['year'].fillna(movie_features['year'].median()).astype(int)
    movie_features = movie_features.fillna('unknown')

    # 범주형 columns str로 변환
    movie_features['genre'] = movie_features['genre'].apply(lambda x: x if isinstance(x, list) else [])
    movie_features['director'] = movie_features['director'].fillna('unknown').astype(str)

    movie_features['year'] = movie_features['year'].fillna(movie_features['year'].median()).astype(int)

    return movie_features

def recommend_all_users(cfg, model, movie_features, batch_size=500, top_k=100, cat_features=None):
    # 1. 대상 유저 ID 리스트 추출
    user_ids = pd.read_csv(to_absolute_path(cfg.data.submission_path))['user'].unique()
    all_recommendations = {}
    
    for i in range(0, len(user_ids), batch_size):
        batch_user_ids = user_ids[i:i + batch_size]

        # 2. 현재 배치의 유저들을 위한 임시 데이터프레임 생성
        df_batch_users = pd.DataFrame(batch_user_ids, columns=['user'])
        
        # 3. Cross Join (유저 배치 x 모든 영화 피처)
        # how='cross'를 사용하면 별도의 key 생성 없이 모든 조합을 만듭니다.
        # 결과: (batch_size * 6807) 행 생성
        temp_df = df_batch_users.merge(movie_features, how='cross')

        # 4. CatBoost Pool 생성
        # 학습 때 사용하지 않은 'user'와 'movie' ID 컬럼은 제외합니다.
        # (주의: movie_features 안에 있는 피처들의 순서가 학습 때와 동일해야 합니다)
        X_test = temp_df.drop(['user', 'item'], axis=1, errors='ignore')
        
        if cat_features is None:
            predict_pool = Pool(data=X_test)
        else:
            predict_pool = Pool(data=X_test, cat_features=cat_features)

        # 4. 점수 예측
        temp_df['score'] = model.predict(predict_pool)
        
        # 5. 유저별 상위 K개 추출 (메모리 절약을 위해 모든 점수를 다 들고 있지 않음)
        for u_id in batch_user_ids:
            user_scores = temp_df[temp_df['user'] == u_id][['item', 'score']]
            top_movies = user_scores.nlargest(top_k, 'score')
            all_recommendations[u_id] = top_movies['item'].tolist()
            
        print(f"Progress: {i + len(batch_user_ids)} / {len(user_ids)} users processed")
        
    return all_recommendations

def save_to_csv_long(recommendations, save_path):
    rows = []
    for user_id, movie_list in recommendations.items():
        for movie_id in movie_list:
            rows.append([user_id, movie_id])
            
    df_long = pd.DataFrame(rows, columns=['user', 'item'])
    df_long.to_csv(save_path, index=False)
    print(f"Saved long-format file to: {save_path}")


@hydra.main(version_base=None, config_path="../../config/CatBoost", config_name="config")
def main(cfg: DictConfig) -> None:
    # Load paths from config
    model_path = to_absolute_path(cfg.inference.model_path)
    model = CatBoostRanker()
    model.load_model(model_path)
    
    movie_features = load_movie_features(cfg)

    # 범주형 피처 인덱스 정의 (장르, 감독 등)
    requested_cat_features = ['genre', 'director']
    cat_features = [c for c in requested_cat_features if c in movie_features.columns]

    recommendations = recommend_all_users(
        cfg=cfg,
        model=model,
        movie_features=movie_features,
        batch_size=int(cfg.inference.batch_size),
        top_k=int(cfg.inference.top_k),
        cat_features=cat_features,
    )

    save_to_csv_long(recommendations, "recommendation_list.csv")


if __name__ == "__main__":
    main()