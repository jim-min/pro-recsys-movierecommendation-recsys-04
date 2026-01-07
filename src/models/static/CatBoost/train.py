import pandas as pd
import numpy as np
from catboost import CatBoostRanker, Pool

import os
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm
import json

def create_train_set(df_interaction, df_movie, n_negative=10):
    all_movies = df_movie['item'].unique()
    train_data = []
    for user, group in tqdm(df_interaction.groupby('user'), total=df_interaction['user'].nunique()):
        # Positive: 유저가 실제 본 영화
        pos_movies = group['item'].values
        n_positive = len(pos_movies)
        
        # Negative 샘플 수 조정: (n_positive + n_negative) <= 1023
        max_negatives = min(1023 - n_positive, n_negative * n_positive)
        if max_negatives <= 0:
            # 양성 샘플이 너무 많은 경우, 일부만 사용
            pos_movies = pos_movies[:1023]
            n_positive = len(pos_movies)
            max_negatives = 1023 - n_positive
        
        for m in pos_movies:
            train_data.append([user, m, 1])
        
        # Negative: 안 본 영화 중 max_negatives 개수만큼 샘플링
        unwatched = np.setdiff1d(all_movies, pos_movies)
        if len(unwatched) > 0:
            neg_samples = np.random.choice(
                unwatched, 
                size=min(len(unwatched), max_negatives), 
                replace=False
            )
            for m in neg_samples:
                train_data.append([user, m, 0])
            
    return pd.DataFrame(train_data, columns=['user', 'item', 'label'])


@hydra.main(version_base=None, config_path="../../config/CatBoost", config_name="config")
def main(cfg: DictConfig) -> None:
    np.random.seed(int(cfg.train.random_seed))

    interaction_path = to_absolute_path(cfg.data.interaction_path)
    title_path = to_absolute_path(cfg.data.title_path)
    director_path = to_absolute_path(cfg.data.directors_path)
    year_path = to_absolute_path(cfg.data.year_path)
    model_path = to_absolute_path(cfg.train.model_path)

    # [1] 데이터 로드 (예시)
    # df_interaction: user, movie (implicit feedback)
    # df_title: movie, title
    # df_year: movie, year
    # df_director: movie, director
    # df_genre: movie, genre_list
    df_interaction = pd.read_csv(interaction_path)

    df_title = pd.read_csv(title_path, sep="\t")
    df_year = pd.read_csv(year_path, sep="\t")
    df_year['item'] = df_year['item'].astype(int)
    
    df_directors = pd.read_csv(director_path, sep="\t")
    # 한 영화에 감독이 여러 명일 경우 하나로 합침 (중복 행 방지)
    df_directors = df_directors.groupby('item')['director'].apply(lambda x: ", ".join(x)).reset_index()
    df_directors['item'] = df_directors['item'].astype(int)

    with open(cfg.data.genre_path, 'r') as f:
        genre_dict = json.load(f)
    df_genre = pd.DataFrame(list(genre_dict.items()), columns=['item', 'genre'])
    df_genre['item'] = df_genre['item'].astype(int)

    # 1:10 비율로 학습 데이터 생성
    train_df = create_train_set(df_interaction, df_title, n_negative=cfg.train.n_negative)

    # [2] 메타데이터 결합
    df_movie_meta = df_year.merge(df_directors, on='item', how='left') \
                           .merge(df_genre, on='item', how='left')

    # 훈련 데이터셋에 영화 메타데이터 결합
    # concat 대신 merge를 써야 user-item 쌍에 맞는 영화 정보가 들어감
    train_df = train_df.merge(df_movie_meta, on='item', how='left')

    # 중복 제거 (불필요할 수 있으나 안전을 위해 수행)
    train_df = train_df.drop_duplicates(subset=['user', 'item'], keep='first')

    # [3] 유저별 정렬 (YetiRank의 필수 조건)
    train_df = train_df.sort_values(by='user').reset_index(drop=True)

    # [4] Group ID 생성 및 라벨 분리
    group_id = train_df['user'].values  # CatBoost가 유저를 식별하는 기준
    y_train = train_df['label'].values
    X_train = train_df.drop(['user', 'item', 'label'], axis=1)

    # 범주형 columns str로 변환
    for col in X_train.columns:
        if col in ['genre', 'director']:
            X_train[col] = X_train[col].fillna('unknown')
            X_train[col] = X_train[col].astype(str)

        elif col in ['year']:
            X_train[col] = X_train[col].fillna(X_train[col].median())
            X_train[col] = X_train[col].astype(float)

    # 범주형 피처 인덱스 정의 (장르, 감독 등)
    requested_cat_features = ['genre', 'director']
    cat_features = [c for c in requested_cat_features if c in X_train.columns]

    # [5] CatBoost 전용 데이터 객체 Pool 생성
    train_pool = Pool(
        data=X_train,
        label=y_train,
        group_id=group_id,
        cat_features=cat_features,
    )

    # [6] 모델 정의
    model = CatBoostRanker(
        iterations=int(cfg.train.iterations),
        learning_rate=float(cfg.train.learning_rate),
        depth=int(cfg.train.depth),
        loss_function=str(cfg.train.loss_function),
        eval_metric=str(cfg.train.eval_metric),
        task_type=str(cfg.train.task_type),
        verbose=int(cfg.train.verbose),
        random_seed=int(cfg.train.random_seed),
    )

    # [7] 학습 시작
    model.fit(train_pool)

    # [8] 모델 저장
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)


if __name__ == "__main__":
    main()