import os
import numpy as np
import torch
from torch.optim import Adam


def bpr_loss(user_vec, pos_vec, neg_vec):
    pos_score = torch.sum(user_vec * pos_vec, dim=1)
    neg_score = torch.sum(user_vec * neg_vec, dim=1)
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-12).mean()


class LightGCNTrainer:
    def __init__(
        self,
        model,
        norm_adj,
        user_pos_list,
        num_items,
        lr,
        weight_decay,
        device,
        ckpt_path="best.pt",
        early_stop_patience=20,   # ✅ early stopping
    ):
        self.model = model.to(device)
        self.norm_adj = norm_adj.to(device)

        self.user_pos_list = user_pos_list
        self.user_pos_set = [set(p) for p in user_pos_list]

        self.num_items = num_items
        self.device = device
        self.ckpt_path = ckpt_path

        self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.best_loss = float("inf")
        self.no_improve_cnt = 0
        self.early_stop_patience = early_stop_patience

    def sample_batch(self, batch_size):
        users = np.random.randint(0, len(self.user_pos_list), size=batch_size)

        pos_items, neg_items = [], []
        for u in users:
            pos = np.random.choice(self.user_pos_list[u])
            while True:
                neg = np.random.randint(0, self.num_items)
                if neg not in self.user_pos_set[u]:
                    break
            pos_items.append(pos)
            neg_items.append(neg)

        return (
            torch.tensor(users, device=self.device),
            torch.tensor(pos_items, device=self.device),
            torch.tensor(neg_items, device=self.device),
        )

    def train(self, epochs, batch_size, steps_per_epoch):
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for _ in range(steps_per_epoch):
                u, p, n = self.sample_batch(batch_size)
                user_emb, item_emb = self.model(self.norm_adj)
                loss = bpr_loss(user_emb[u], item_emb[p], item_emb[n])

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss.item()

            avg_loss = total_loss / steps_per_epoch
            print(f"[Epoch {epoch}] BPR loss = {avg_loss:.6f}")

            # ===== best model save =====
            if avg_loss < self.best_loss - 1e-5:
                self.best_loss = avg_loss
                self.no_improve_cnt = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "best_loss": self.best_loss,
                    },
                    self.ckpt_path,
                )
                print(f"💾 Best model updated (loss={avg_loss:.6f})")

            else:
                self.no_improve_cnt += 1
                print(
                    f"⏳ No improvement ({self.no_improve_cnt}/{self.early_stop_patience})"
                )

                if self.no_improve_cnt >= self.early_stop_patience:
                    print("🛑 Early stopping triggered")
                    break
