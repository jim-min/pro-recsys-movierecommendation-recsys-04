import torch
from torch.utils.data import DataLoader
import numpy as np

# 모듈 임포트 (디렉토리 구조에 맞게 수정)
from src.data.preprocessor import DataProcessor
from src.data.dataset import StaticDataset
from src.models.multi_vae import MultiVAE
from src.models.ease import Ease
from src.models.ea_vae import EAVAE
from src.loss.loss import multivae_loss
from trainer import Trainer

# --- 설정 (Config) ---
CONFIG = {
    'input_dim': None, # 데이터 로드 후 설정됨
    'q_dims': [1024, 256],
    'dropout_rate': 0.2,
    
    'batch_size': 500,
    'epochs': 300,
    'lr': 2e-4,
    'patience': 100,
    'k': 10, # Recall@10
    
    'total_anneal_steps': 50000,
    'anneal_cap': 0.2,
    
    'save_dir': './saved/models/ea-vae/',
    'data_path': './data/train/train_ratings.csv',
    
    'seed': 87,
    
    'model': 'EAVAE',
}

def main():
    # 1. Data Processing
    print("Loading Data...")
    processor = DataProcessor(CONFIG['data_path'])
    df = processor.load_and_process()
    
    # 1-1. Split (Train/Valid) - 비율 보장형 or Leave-One-Out
    # 여기서는 안전한 비율 보장형 사용
    train_mat, valid_mat = processor.split_data(df, test_ratio=0.01, seed=CONFIG['seed'])
    
    if CONFIG['model'] == 'MultiVAE':
        num_users, num_items = train_mat.shape
        CONFIG['input_dim'] = num_items
        print(f"Users: {num_users}, Items: {num_items}")

        # 2. Dataset & DataLoader
        train_dataset = StaticDataset(train_mat)
        valid_dataset = StaticDataset(train_mat, valid_mat) # Valid는 Input(Train), Target(Valid) 둘 다 필요
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=True
        )
        
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=False
        )
        
        # 3. Model Build
        q_dims = [CONFIG['input_dim'], *CONFIG['q_dims']]
        model = MultiVAE(q_dims=q_dims, dropout_rate=CONFIG['dropout_rate'])
        
        # 4. Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
        
        # 5. Trainer Init & Run
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_function=multivae_loss,
            train_loader=train_loader,
            valid_loader=valid_loader,
            config=CONFIG
        )
        
        trainer.fit()
        
    elif CONFIG['model'] in ['EASE', 'EAVAE']:
        model = Ease(reg_lambda=500)
        model.fit(train_mat)
        
        valid_dataset = StaticDataset(train_mat, valid_mat)
        
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=False
        )
        
        recall_list = []
        # 설정값 가져오기
        k = CONFIG['k']
        device = model.weight.device  # 모델이 있는 GPU로 데이터 이동
        
        print(f"Calculating Validation Recall@{k} on {device}...")


        # 평가 모드 (Gradient 계산 비활성화 -> 메모리 절약 & 속도 향상)
        with torch.no_grad():
            for input_data, target_data in valid_loader:
                # 1. 데이터를 GPU로 이동
                # input_data: Train 데이터 (모델 입력 & 이미 본 아이템 마스킹용)
                # target_data: Valid 데이터 (정답지)
                input_data = input_data.to(device)
                target_data = target_data.to(device)
                
                # 2. 모델 예측 (Inference)
                # remove_seen=True: input_data(학습때 본거)는 추천 리스트에서 제외 (-inf 처리)
                # DataLoader에서 나왔으므로 input_data는 이미 Tensor 형태입니다.
                batch_scores = model.predict(input_data, k=k, remove_seen=True)
                
                # 3. Top-K 추출
                # batch_scores는 (Batch_Size, Items) 크기
                _, topk_indices = torch.topk(batch_scores, k=k, dim=1)
                
                # 4. Recall 계산 (Batch 단위)
                # gather: 정답지(target)에서 우리가 예측한(topk) 위치의 값이 1인지 0인지 확인
                # 결과 hits: (Batch_Size, k)
                hits = torch.gather(target_data, 1, topk_indices)
                
                # 분자: 맞춘 개수 (유저별 합계)
                hit_count = hits.sum(dim=1)
                
                # 분모: 실제 정답 개수 (유저별 합계)
                total_true = target_data.sum(dim=1)
                
                # 0으로 나누기 방지 (정답이 0개인 유저는 분모를 1로 처리하되 결과는 0)
                recall_batch = hit_count / (total_true + 1e-9)
                
                # 결과 리스트에 저장 (CPU로 내려서 numpy 변환)
                recall_list.append(recall_batch.cpu().numpy())

        # 5. 최종 평균 Recall 계산
        all_recalls = np.concatenate(recall_list)
        mean_recall = np.mean(all_recalls)
        
        print(f"Final Validation Recall@{k}: {mean_recall:.4f}")
        
        if CONFIG['model'] == 'EAVAE':
            num_users, num_items = train_mat.shape
            CONFIG['input_dim'] = num_items
            print(f"Users: {num_users}, Items: {num_items}")
            
            train_dataset = StaticDataset(train_mat)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=CONFIG['batch_size'], 
                shuffle=True
            )
            
            q_dims = [CONFIG['input_dim'], *CONFIG['q_dims']]
            vae_model = EAVAE(ease_weight=model.weight, q_dims=q_dims, dropout_rate=CONFIG['dropout_rate'], ease_rate=0.3)
            
            # 4. Optimizer
            optimizer = torch.optim.Adam(vae_model.parameters(), lr=CONFIG['lr'])
        
            # 5. Trainer Init & Run
            trainer = Trainer(
                model=vae_model,
                optimizer=optimizer,
                loss_function=multivae_loss,
                train_loader=train_loader,
                valid_loader=valid_loader,
                config=CONFIG
            )
        
            trainer.fit()

if __name__ == "__main__":
    main()