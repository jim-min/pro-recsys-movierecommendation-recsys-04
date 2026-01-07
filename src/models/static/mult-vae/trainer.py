import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from metrics import recall_at_k  # valid 있을 때만 사용
from recommend import recommend_topk


class MultiVAETrainer:
    def __init__(
        self,
        model,
        train_mat,
        valid_gt,
        num_items,
        lr,
        weight_decay,
        device,
        ckpt_path,
        early_stop_patience=20,
        kl_max_weight=0.2,
        kl_anneal_steps=20000,
        # ===== New: No-valid early stop knobs =====
        no_valid_window=8,          # 최근 window epoch 기준으로 "감소율 정체" 판단
        no_valid_eps=1e-3,          # 곡률(감소율 변화) 임계값: 작을수록 더 엄격
        kl_dom_alpha=1.0,           # (beta*kl) > recon*alpha 되면 "KL 지배 시작" 신호
        no_valid_min_epochs=30,     # 너무 초반에 멈추지 않게 하한
        stop_needs_both=True,       # True면 (정체 AND KL지배) 둘 다 만족할 때만 stop
        save_every_epoch=False,     # 제출 제한이면 False 추천(최종 best 1개만 저장)
    ):
        self.model = model.to(device)
        self.train_mat = train_mat
        self.valid_gt = valid_gt
        self.num_items = num_items
        self.device = device
        self.ckpt_path = ckpt_path

        self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # valid 있을 때만 의미 있음(그래도 유지)
        self.scheduler = ReduceLROnPlateau(
            self.optim, mode="max", factor=0.5, patience=15
        )


        self.early_stop_patience = early_stop_patience
        self.best_score = -1.0
        self.no_improve = 0

        self.kl_max_weight = kl_max_weight
        self.kl_anneal_steps = kl_anneal_steps
        self.global_step = 0

        # ===== New fields =====
        self.no_valid_window = int(no_valid_window)
        self.no_valid_eps = float(no_valid_eps)
        self.kl_dom_alpha = float(kl_dom_alpha)
        self.no_valid_min_epochs = int(no_valid_min_epochs)
        self.stop_needs_both = bool(stop_needs_both)
        self.save_every_epoch = bool(save_every_epoch)

        # 기록용 (no valid일 때 early stop 판단)
        self.loss_hist = []       # epoch loss
        self.recon_hist = []      # epoch recon
        self.kl_hist = []         # epoch kl
        self.beta_hist = []       # epoch beta
        self.kldom_hist = []      # epoch (beta*kl)/recon ratio

        # no-valid 상황에서 "best"를 뭘로 둘지: heuristic score로 관리
        # (loss 절댓값 대신, "붕괴 전" 신호를 반영한 proxy를 쓰자)
        self.best_proxy = -1e18

    def _kl_weight(self):
        # 기존과 동일 (선형 anneal)
        return min(self.kl_max_weight, self.kl_max_weight * self.global_step / self.kl_anneal_steps)

    @staticmethod
    def _safe_div(a, b, eps=1e-12):
        return a / (b + eps)

    def _no_valid_should_stop(self, epoch):
        """
        Loss curvature(감소율 정체) + KL dominance(규제 지배) 조합
        - plateau: 최근 window에서 loss 감소율 변화(곡률)가 매우 작아짐
        - kl_dom: beta*kl 항이 recon에 근접/우세해지기 시작
        """
        if epoch < self.no_valid_min_epochs:
            return False, {"plateau": False, "kl_dom": False, "reason": "min_epochs"}

        w = self.no_valid_window
        if len(self.loss_hist) < (2 * w + 1):
            return False, {"plateau": False, "kl_dom": False, "reason": "warmup_window"}

        # ---- 1) Loss curvature (감소율 정체) ----
        # 최근 w epoch 평균 감소량 vs 그 이전 w epoch 평균 감소량 비교
        # d_recent, d_prev가 거의 같아지면 "학습이 더 이상 유의미하게 변하지 않는다" 신호
        loss_now = np.mean(self.loss_hist[-w:])
        loss_prev = np.mean(self.loss_hist[-2*w:-w])
        loss_prev2 = np.mean(self.loss_hist[-3*w:-2*w])

        d_recent = loss_prev - loss_now     # 최근 window 동안 얼마나 내려갔나
        d_prev = loss_prev2 - loss_prev     # 그 이전 window 동안 얼마나 내려갔나
        curvature = abs(d_recent - d_prev)

        plateau = curvature < self.no_valid_eps

        # ---- 2) KL dominance ----
        # beta*kl 이 recon에 가까워지거나 넘어서기 시작하면, latent 수축/표현력 붕괴 신호
        recon_now = float(self.recon_hist[-1])
        beta_now = float(self.beta_hist[-1])
        kl_now = float(self.kl_hist[-1])

        kl_term = beta_now * kl_now
        ratio = self._safe_div(kl_term, recon_now)  # (beta*kl)/recon

        kl_dom = kl_term > recon_now * self.kl_dom_alpha

        # ---- stop rule ----
        if self.stop_needs_both:
            should_stop = plateau and kl_dom
        else:
            should_stop = plateau or kl_dom

        info = {
            "plateau": plateau,
            "kl_dom": kl_dom,
            "curvature": float(curvature),
            "d_recent": float(d_recent),
            "d_prev": float(d_prev),
            "kl_term": float(kl_term),
            "recon": float(recon_now),
            "ratio": float(ratio),
            "reason": "both" if self.stop_needs_both else "either",
        }
        return should_stop, info

    def _proxy_score(self, epoch):
        """
        valid 없을 때 best를 고르기 위한 proxy.
        목표: '붕괴 전'의 안정적인 지점 선호.
        - loss가 너무 낮게만 가는 모델(=과적합/붕괴) 피하기 위해,
          KL dominance ratio가 커지면 페널티.
        """
        loss = float(self.loss_hist[-1])
        ratio = float(self.kldom_hist[-1])  # (beta*kl)/recon

        # loss는 낮을수록 좋지만, ratio가 커지는 순간부터는 위험해짐
        # 그래서 ratio에 페널티를 줘서 "적당히"에서 best를 잡게 함.
        # (상수는 안전하게 보수적으로 잡음)
        proxy = (-loss) - 0.5 * max(0.0, ratio - 0.8)
        return proxy

    def train(self, epochs, batch_size, topk=10):
        num_users = self.train_mat.shape[0]

        for epoch in range(1, epochs + 1):
            self.model.train()
            perm = np.random.permutation(num_users)

            total_loss = 0.0
            total_recon = 0.0
            total_kl = 0.0
            step_count = 0

            for start in range(0, num_users, batch_size):
                batch_users = perm[start : start + batch_size]
                x = torch.from_numpy(self.train_mat[batch_users].toarray()).float().to(self.device)

                logits, mu, logvar = self.model(x)
                log_softmax = F.log_softmax(logits, dim=1)

                # recon: cross-entropy with implicit feedback
                recon = -(log_softmax * x).sum(dim=1).mean()

                # KL
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

                beta = self._kl_weight()
                loss = recon + beta * kl

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += float(loss.item())
                total_recon += float(recon.item())
                total_kl += float(kl.item())
                step_count += 1

                self.global_step += 1

            # epoch 평균
            avg_loss = total_loss / max(1, step_count)
            avg_recon = total_recon / max(1, step_count)
            avg_kl = total_kl / max(1, step_count)
            beta_epoch = float(self._kl_weight())
            ratio_epoch = float(self._safe_div(beta_epoch * avg_kl, avg_recon))

            self.loss_hist.append(avg_loss)
            self.recon_hist.append(avg_recon)
            self.kl_hist.append(avg_kl)
            self.beta_hist.append(beta_epoch)
            self.kldom_hist.append(ratio_epoch)

            # --- Validation Section (있으면 사용) ---
            has_valid = any(len(v) > 0 for v in self.valid_gt.values())

            if has_valid:
                rec = recommend_topk(self.model, self.train_mat, topk, self.device, batch_size)
                actual, pred = [], []
                for u in range(num_users):
                    if len(self.valid_gt[u]) > 0:
                        actual.append(self.valid_gt[u])
                        pred.append(rec[u].tolist())
                current_score = recall_at_k(actual, pred, k=topk)

                self.scheduler.step(current_score)

                curr_lr = self.optim.param_groups[0]["lr"]
                print(
                    f"[Epoch {epoch}] loss={avg_loss:.4f} recon={avg_recon:.4f} kl={avg_kl:.4f} "
                    f"beta={beta_epoch:.4f} (beta*kl)/recon={ratio_epoch:.3f} "
                    f"Val Recall@{topk}={current_score:.4f}, lr={curr_lr:.6f}"
                )

                # valid 있으면 기존 방식대로 best 저장
                if current_score > self.best_score + 1e-6:
                    self.best_score = current_score
                    self.no_improve = 0
                    torch.save({"model_state": self.model.state_dict(), "epoch": epoch}, self.ckpt_path)
                    print("💾 Best model updated (by valid)")
                else:
                    self.no_improve += 1
                    if self.no_improve >= self.early_stop_patience:
                        print(f"🛑 Early stopping at epoch {epoch} (by valid patience)")
                        break

            else:
                # ---- No-valid logging ----
                curr_lr = self.optim.param_groups[0]["lr"]
                print(
                    f"[Epoch {epoch}] loss={avg_loss:.4f} recon={avg_recon:.4f} kl={avg_kl:.4f} "
                    f"beta={beta_epoch:.4f} (beta*kl)/recon={ratio_epoch:.3f} lr={curr_lr:.6f}"
                )

                # ---- No-valid: proxy best save ----
                proxy = self._proxy_score(epoch)

                if self.save_every_epoch:
                    torch.save(
                        {"model_state": self.model.state_dict(), "epoch": epoch, "loss": avg_loss},
                        self.ckpt_path.replace(".pt", f"_epoch{epoch}.pt"),
                    )

                if proxy > self.best_proxy + 1e-9:
                    self.best_proxy = proxy
                    torch.save({"model_state": self.model.state_dict(), "epoch": epoch}, self.ckpt_path)
                    print(f"💾 Best model updated (proxy={proxy:.6f})")

                # ---- No-valid: stop rule (1순위) ----
                should_stop, info = self._no_valid_should_stop(epoch)
                if should_stop:
                    print(
                        "🛑 Early stopping (no-valid heuristic) at epoch "
                        f"{epoch} | plateau={info['plateau']} kl_dom={info['kl_dom']} "
                        f"curv={info.get('curvature', 0):.6f} ratio={(info.get('ratio', 0)):.3f}"
                    )
                    break
