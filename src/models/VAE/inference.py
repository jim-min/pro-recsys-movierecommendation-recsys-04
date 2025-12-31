import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from src.data.preprocessor import DataProcessor
from src.data.dataset import StaticDataset
from src.models.multi_vae import MultiVAE
from src.loss.loss import multivae_loss

CONFIG = {
    # 최적에폭
    'best_epoch': 30,
    
    # 모델 하이퍼 파라미터
    'hidden_dim': 2048,
    'latent_dim': 256,
    'dropout_rate': 0.5,
    'lr': 5e-4,
    'batch_size': 500,
    'k': 10,
    
    # Annealing 설정
    'anneal_cap': 0.2,
    # 'anneal_ratio': 0.7,
    'total_anneal_steps' : 30000,
    
    # 경로 설정
    'data_path': './data/train/train_ratings.csv',
    'output_path': '../output/multi_vae_submission_2.csv'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"🚀 Starting Inference Pipeline on {device}")
    print(f"🎯 Target Epoch: {CONFIG['best_epoch']} (Full Training)")
    
    
    # ==========================================
    # 1. Data Loading & Processing (Full Data)
    # ==========================================
    print("\n[Step 1] Loading  Processing Data...")
    processor = DataProcessor(CONFIG['data_path'])
    
    # Split 없이 전체 데이터 로드 및 인코딩
    df = processor.load_and_process()
    
    # 전체 데이터를 CSR Matrix로 변환
    full_matrix = processor._create_csr_matrix(df)
    print(f"    Data Shape: {full_matrix.shape} (Users: {full_matrix.shape[0]}, Items: {full_matrix.shape[1]})")
    
    # Dataset 생성
    full_dataset = StaticDataset(full_matrix)
    
    # Loader 생성
    # 학습용 Loader(Shuffle=True)
    train_loader = DataLoader(full_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # 추론용 Loader(Shuffle=False) 순서지켜야함
    inference_loader = DataLoader(full_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    
    # ==========================================
    # 2. Model & Optimizer Initialization
    # ==========================================
    print("\n[Step 2] Initializing Model...")
    input_dim = full_matrix.shape[1]
    p_dims = [input_dim, CONFIG['hidden_dim'], CONFIG['latent_dim']]
    
    model = MultiVAE(p_dims=p_dims, dropout_rate=CONFIG['dropout_rate']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    
    # ==========================================
    # 3. Full Training Loop
    # ==========================================
    print("\n[Step 3] Re-training on Full Data...")
    model.train()
    
    # Annealing Step 자동 계산 (배치 수 * 에폭 * 비율)
    total_steps = CONFIG['total_anneal_steps'] #len(train_loader) * CONFIG['best_epoch'] * CONFIG['anneal_ratio']
    update_count = 0
    
    for epoch in range(1, CONFIG['best_epoch'] + 1):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{CONFIG['best_epoch']}", leave=False)
        
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward
            recon_batch, mu, logvar = model(batch)
            
            # Anneal Logic
            if total_steps > 0:
                anneal = min(CONFIG['anneal_cap'], 1.0 * update_count / total_steps)
            else:
                anneal = CONFIG['anneal_cap']
            update_count += 1
            
            # Loss & Backward
            loss = multivae_loss(recon_batch, batch, mu, logvar, anneal)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'anneal': anneal})
        
        print(f"    Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f}")
    
    print("     >>> Full Training Finished!")
    
    
    # ==========================================
    # 4. Inference & Submission Generation
    # ==========================================
    print("\n[Step 4] Generating Submission File...")
    model.eval()
    
    all_users = []
    all_items = []
    
    # User ID 복원을 위한 시작 인덱스
    user_start_idx = 0
    
    with torch.no_grad():
        for batch_input in tqdm(inference_loader, desc="Inferencing"):
            
            batch_input = batch_input.to(device)
            
            # 예측
            recon_batch, _, _ = model(batch_input)
            
            # 이미 본 아이템 마스킹
            recon_batch[batch_input.nonzero(as_tuple=True)] = -float('inf')
            
            # Top-K 선정
            _, topk_indices = torch.topk(recon_batch, k=CONFIG['k'], dim=1)
            
            # CPU & Numpy 변환
            topk_indices = topk_indices.cpu().numpy()
            
            # ID Decoding
            batch_size = batch_input.size(0)
            
            # 현재 배치의 User Index들
            current_user_indices = np.arange(user_start_idx, user_start_idx + batch_size)
            
            # User ID 복원
            decoded_users = processor.user_encoder.inverse_transform(current_user_indices)
            
            # Item ID 복원 및 저장
            for i in range(batch_size):
                rec_item_indices = topk_indices[i]
                rec_item_ids = processor.item_encoder.inverse_transform(rec_item_indices)
                
                for item_id in rec_item_ids:
                    all_users.append(decoded_users[i])
                    all_items.append(item_id)
            
            user_start_idx += batch_size
    
    # ==========================================
    # 5. Save to CSV
    # ==========================================
    submission = pd.DataFrame({
        'user': all_users,
        'item': all_items
    })
    
    submission.to_csv(CONFIG['output_path'], index=False)
    print(f"\n✅ Submission Saved Successfully: {CONFIG['output_path']}")
    print(f"   Total Rows: {len(submission)}")

if __name__ == '__main__':
    main()
    