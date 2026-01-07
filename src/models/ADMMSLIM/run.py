import os
from tqdm import tqdm
import pandas as pd
import hydra
from omegaconf import DictConfig
from preprocessing import load_and_preprocess_data
from model import AdmmSlim
from inference import generate_recommendations

@hydra.main(version_base=None, config_path="../../config/SLIM", config_name="config")
def main(cfg: DictConfig):
    """
    ADMMSLIM 모델을 사용하여 추천 시스템을 실행하는 메인 함수.

    Args:
        cfg (DictConfig): Hydra 설정값을 포함하는 인자. 다음과 같은 정보를 포함해야 합니다:
            - dataset.data_path (str): 데이터셋 경로.
            - model_args.lambda_1 (float): L1 정규화 가중치.
            - model_args.lambda_2 (float): L2 정규화 가중치.
            - model_args.rho (float): 페널티 매개변수.
            - model_args.positive (bool): 계수를 양수로 제한할지 여부.
            - model_args.n_iter (int): 최대 반복 횟수.
            - model_args.verbose (bool): 학습 로그 출력 여부.

    Workflow:
        1. 데이터 로드 및 전처리.
        2. ADMMSLIM 모델 학습.
        3. 사용자별 추천 생성.
        4. 추천 결과를 CSV 파일로 저장.

    Output:
        'recommendations.csv' 파일에 추천 결과를 저장합니다.
    """
    data_path = cfg.dataset.data_path
    years_path = cfg.dataset.years_path
    # 데이터 로드 및 전처리
    valid_ratio = float(cfg.get('split', {}).get('valid_ratio', 0.0))
    split_seed = int(cfg.get('split', {}).get('seed', 42))
    train_df, years_df, train_matrix, valid_matrix, user_encoder, item_encoder = load_and_preprocess_data(
        data_path,
        years_path=years_path,
        valid_ratio=valid_ratio,
        seed=split_seed,
    )

    # 모델 생성 및 학습
    model = AdmmSlim(lambda_1=cfg.model_args.lambda_1, 
                     lambda_2=cfg.model_args.lambda_2, 
                     rho=cfg.model_args.rho, 
                     positive=cfg.model_args.positive, 
                     n_iter=cfg.model_args.n_iter, 
                     verbose=cfg.model_args.verbose)
    eval_every = int(cfg.get('validation', {}).get('eval_every', 10))
    eval_k = int(cfg.get('validation', {}).get('k', 10))
    model.fit(train_matrix, valid_X=valid_matrix, eval_every=eval_every, eval_k=eval_k)

    model_path = cfg.get("output", {}).get("model_path", None)
    if model_path:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        model.save(model_path)

    # 추천 생성
    user_recommendations = generate_recommendations(
        model, 
        train_matrix, 
        train_df['user_id'].nunique(), 
        K=int(cfg.get('inference', {}).get('topk', 10)), 
        future_delete=cfg.inference.future_delete, 
        years=years_df, 
        user_encoder=user_encoder, 
        item_encoder=item_encoder
    )

    # 추천 결과 저장
    recommendations = []

    for user_id, item_ids in tqdm(user_recommendations.items(), desc="Saving Recommendations"):
        user_original_id = user_encoder.inverse_transform([user_id])[0]
        for item_id in item_ids:
            item_original_id = item_encoder.inverse_transform([item_id])[0]
            recommendations.append({
                'user': user_original_id,
                'item': item_original_id
            })

    recommendations_df = pd.DataFrame(recommendations)
    recommendations_path = cfg.get("output", {}).get("recommendations_path", "recommendations.csv")
    os.makedirs(os.path.dirname(recommendations_path) or ".", exist_ok=True)
    recommendations_df.to_csv(recommendations_path, index=False)

if __name__ == '__main__':
    main()