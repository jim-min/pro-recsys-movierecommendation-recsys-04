import torch
import numpy as np
from torch.optim import Adam
from metrics import recall_at_k
from recommend import recommend_topk

class FMTrainer:
    def __init__(
        self, model, train_mat, valid_user_pos, num_items, 
        item_attr_mat, item_attr_mask, user_offset, item_offset,
        lr, weight_decay, device, ckpt_path, early_stop_patience=20
    ):
        """
        [수정사항] run.py에서 넘겨주는 'train_mat' 및 모든 인자를 받도록 설계됨
        """
        self.model = model.to(device)
        self.train_mat = train_mat  # 학습용 Sparse Matrix
        self.valid_user_pos = valid_user_pos  # 검증용 정답 데이터
        self.num_items = num_items
        self.item_attr_mat = item_attr_mat.to(device)
        self.item_attr_mask = item_attr_mask.to(device)
        self.user_offset = user_offset
        self.item_offset = item_offset
        self.device = device
        self.ckpt_path = ckpt_path
        
        self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.early_stop_patience = early_stop_patience
        self.best_score = -1.0
        self.no_improve = 0

    def sample(self, batch_size):
        """Positive와 Negative 샘플을 추출"""
        num_users = self.train_mat.shape[0]
        users = np.random.randint(0, num_users, size=batch_size)
        pos_items = []
        neg_items = []

        for u in users:
            # 해당 유저가 본 아이템들 (CSR matrix 활용)
            pos_for_u = self.train_mat.indices[self.train_mat.indptr[u] : self.train_mat.indptr[u+1]]
            
            if len(pos_for_u) > 0:
                pos_items.append(np.random.choice(pos_for_u))
            else:
                # 본 아이템이 없으면 랜덤하게 하나 선택 (예외 처리)
                pos_items.append(np.random.randint(0, self.num_items))
            
            # Negative Sampling (안 본 아이템 찾기)
            while True:
                neg = np.random.randint(0, self.num_items)
                if neg not in pos_for_u:
                    neg_items.append(neg)
                    break
                    
        return torch.LongTensor(users).to(self.device), \
               torch.LongTensor(pos_items).to(self.device), \
               torch.LongTensor(neg_items).to(self.device)

    def build_feat(self, u, i):
        """FM 입력을 위한 피처 인덱스와 마스크를 생성"""
        # [User ID, Item ID, Attr1, Attr2, ...]
        batch_size = u.size(0)
        
        u_idx = (u + self.user_offset).unsqueeze(1)
        i_idx = (i + self.item_offset).unsqueeze(1)
        
        # 아이템 속성 가져오기
        attr_idx = self.item_attr_mat[i]
        attr_mask = self.item_attr_mask[i]
        
        # 피처 결합: (Batch, 2 + max_attrs)
        feat_idx = torch.cat([u_idx, i_idx, attr_idx], dim=1)
        # 마스크 결합 (User/Item은 항상 1.0)
        feat_mask = torch.cat([
            torch.ones((batch_size, 2), device=self.device),
            attr_mask
        ], dim=1)
        
        return feat_idx, feat_mask

    def train(self, epochs, batch_size, steps_per_epoch, topk=10):
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            
            for _ in range(steps_per_epoch):
                u, p, n = self.sample(batch_size)
                
                # Positive/Negative 샘플에 대한 피처 구축
                p_idx, p_mask = self.build_feat(u, p)
                n_idx, n_mask = self.build_feat(u, n)
                
                # BPR Loss 계산
                p_score = self.model(p_idx, p_mask)
                n_score = self.model(n_idx, n_mask)
                
                loss = -torch.log(torch.sigmoid(p_score - n_score) + 1e-12).mean()
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                epoch_loss += loss.item()

            # --- Validation (대회 공식 Recall@K 기반) ---
            self.model.eval()
            with torch.no_grad():
                # recommend_topk를 이용해 전체 추천 리스트 생성
                rec = recommend_topk(
                    self.model, self.train_mat, self.item_attr_mat, 
                    self.item_attr_mask, topk, self.device, 
                    self.user_offset, self.item_offset
                )
                
                actual, pred = [], []
                for u_idx, items in self.valid_user_pos.items():
                    if len(items) > 0:
                        actual.append(items)
                        pred.append(rec[u_idx].tolist())
                
                current_recall = recall_at_k(actual, pred, k=topk)
                print(f"[Epoch {epoch}] Loss: {epoch_loss/steps_per_epoch:.4f} | Val Recall@{topk}: {current_recall:.4f}")

            # Early Stopping 체크
            if current_recall > self.best_score + 1e-6:
                self.best_score = current_recall
                self.no_improve = 0
                torch.save(self.model.state_dict(), self.ckpt_path)
                print("💾 Best model saved!")
            else:
                self.no_improve += 1
                if self.no_improve >= self.early_stop_patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    break