import torch
import numpy as np
import os
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, loss_function, train_loader, valid_loader, config):
        """
        Trainer
        
        Args:
            model: VAE 모델
            optimizer: Optimizer (Adam 등) 
            loss_function: VAE Loss 함수
            train_loader: 학습용 DataLoader
            valid_loader: 검증용 DataLoader
            config: 설정 값
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_function = loss_function
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        # Hyper parameters
        self.k = config.get('k', 10)    # Recall@K의 K
        self.epochs = config.get('epochs', 100)
        self.patience = config.get('patience', 10)
        self.save_dir = config.get('save_dir', './saved/models')
        
        # KL Annealing Parameters
        self.total_anneal_steps = config.get('total_anneal_steps', 200000)
        self.anneal_cap = config.get('anneal_cap', 0.2) # Beta의 최댓값
        self.update_count = 0   # 현재 Step count
        
        # Early Stopping State
        self.best_metric = -np.inf
        self.patience_counter = 0
        
        os.makedirs(self.save_dir, exist_ok=True)
        
    def fit(self):
        print(f"Start Training on {self.device}...")
        
        for epoch in range(1, self.epochs + 1):
            # train step
            train_loss = self._train_epoch(epoch)
            
            # valid step
            recall_score, normalized_recall_score = self._validate_epoch(epoch)
            
            # log
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Recall@{self.k}: {recall_score:.4f} | Norm Recall@{self.k}: {normalized_recall_score:.4f}")
            
            # Early stopping & save
            if recall_score > self.best_metric:
                self.best_metric = recall_score
                self.patience_counter = 0
                self.save_checkpoint("best_model.pth")
                print(f"    >>> Best Recall Updated! Model Saved.")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"    >>> Early Stopping Triggered at Epoch {epoch}")
                    break
                
        print('Training Finished')
    
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        # tqdm으로 진행률 표시
        pbar = tqdm(self.train_loader, desc=f"Train Ep {epoch}", leave=False)
        
        for batch in pbar:
            batch = batch.to(self.device)   # (Batch, Items)
            
            # optimizer 초기화
            self.optimizer.zero_grad()
            
            # forward
            recon_batch, mu, logvar = self.model(batch)
            
            # Annealing Factor 계산
            if self.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, 1.0 * self.update_count / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap
            
            self.update_count += 1
            
            # loss 계산
            loss = self.loss_function(recon_batch, batch, mu, logvar, anneal)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'anneal': anneal})
        
        return total_loss / len(self.train_loader)

    def _validate_epoch(self, epoch):
        self.model.eval()
        total_recall = 0.0
        total_normalized_recall = 0.0
        count = 0
        
        with torch.no_grad():
            for input_data, target_data in self.valid_loader:
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)
                
                # forward
                recon_batch, _, _ = self.model(input_data)
                
                # masking
                # 학습 때 본 아이템(input_data가 1인 곳)은 추천 대상에서 제외
                recon_batch[input_data.nonzero(as_tuple=True)] = -float('inf')
                
                # top-k 선택
                # logit 상태에서도 topk 선택
                _, topk_indices = torch.topk(recon_batch, k=self.k, dim=1)
                
                # Recall @ K 계산
                hits = torch.gather(target_data, 1, topk_indices)
                
                batch_recall = hits.sum(dim=1) / target_data.sum(dim=1).clamp(min=1)
                
                # normalized Recall Calculation
                normalized_denominator = target_data.sum(dim=1).clamp(min=1, max=self.k)
                normalized_batch_recall = hits.sum(dim=1) / normalized_denominator
                
                total_recall += batch_recall.sum().item()
                total_normalized_recall += normalized_batch_recall.sum().item()
                
                count += input_data.size(0)
                
        return (total_recall / count, total_normalized_recall / count)
    
    def save_checkpoint(self, filename):
        path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), path)
            