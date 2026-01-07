import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau # 1. 스케줄러 임포트
from metrics import recall_at_k
from recommend import recommend_topk

class MultiVAETrainer:
    def __init__(
        self, model, train_mat, valid_gt, num_items, lr, weight_decay,
        device, ckpt_path, early_stop_patience=20, kl_max_weight=0.2, kl_anneal_steps=20000
    ):
        self.model = model.to(device)
        self.train_mat = train_mat
        self.valid_gt = valid_gt
        self.num_items = num_items
        self.device = device
        self.ckpt_path = ckpt_path
        self.optim = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # 2. 스케줄러 정의: Recall이 7번(patience) 안 오르면 lr을 0.5배(factor)로 감소
        self.scheduler = ReduceLROnPlateau(
            self.optim, mode='max', factor=0.5, patience=15, verbose=True
        )

        self.early_stop_patience = early_stop_patience
        self.best_score = -1.0
        self.no_improve = 0
        self.kl_max_weight = kl_max_weight
        self.kl_anneal_steps = kl_anneal_steps
        self.global_step = 0

    def _kl_weight(self):
        return min(self.kl_max_weight, self.kl_max_weight * self.global_step / self.kl_anneal_steps)

    def train(self, epochs, batch_size, topk=10):
        num_users = self.train_mat.shape[0]

        for epoch in range(1, epochs + 1):
            self.model.train()
            perm = np.random.permutation(num_users)
            total_loss = 0.0

            for start in range(0, num_users, batch_size):
                batch_users = perm[start : start + batch_size]
                x = torch.from_numpy(self.train_mat[batch_users].toarray()).float().to(self.device)

                logits, mu, logvar = self.model(x)
                log_softmax = F.log_softmax(logits, dim=1)
                recon = -(log_softmax * x).sum(dim=1).mean()
                
                # KL Divergence
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                loss = recon + self._kl_weight() * kl

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss.item()
                self.global_step += 1

            # --- Validation Section ---
            has_valid = any(len(v) > 0 for v in self.valid_gt.values())
            
            if has_valid:
                rec = recommend_topk(self.model, self.train_mat, topk, self.device, batch_size)
                actual, pred = [], []
                for u in range(num_users):
                    if len(self.valid_gt[u]) > 0:
                        actual.append(self.valid_gt[u])
                        pred.append(rec[u].tolist())
                current_score = recall_at_k(actual, pred, k=topk)
                
                # 3. 스케줄러 스텝: Recall 점수를 기준으로 lr 조정 여부 판단
                self.scheduler.step(current_score)
                
                # 현재 lr을 확인하기 위해 로그에 추가
                curr_lr = self.optim.param_groups[0]['lr']
                print(f"[Epoch {epoch}] loss={total_loss:.4f}, Val Recall@{topk}={current_score:.4f}, lr={curr_lr:.6f}")
            else:
                current_score = -total_loss
                print(f"[Epoch {epoch}] loss={total_loss:.4f}")

            if current_score > self.best_score + 1e-6:
                self.best_score = current_score
                self.no_improve = 0
                torch.save({"model_state": self.model.state_dict(), "epoch": epoch}, self.ckpt_path)
                print("💾 Best model updated")
            else:
                self.no_improve += 1
                if self.no_improve >= self.early_stop_patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    break