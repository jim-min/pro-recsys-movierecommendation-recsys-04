import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


class S3RecPretrainTrainer:
    def __init__(
        self,
        model,
        train_loader,
        device,
        lr=1e-3,
        weight_decay=0.0,
        output_dir="./output/S3RecPretrain/",
        use_amp=True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # masked positions만 loss 계산 (labels가 -100인 곳은 무시)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.use_amp = use_amp and (device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

    def train(self, epochs=50, log_every=200):
        best_loss = 1e18
        best_path = os.path.join(self.output_dir, "best.pt")

        for ep in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()

            total_loss = 0.0
            for step, batch in enumerate(self.train_loader, start=1):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                pad_mask = batch["pad_mask"].to(self.device)

                self.optim.zero_grad(set_to_none=True)

                with autocast(enabled=self.use_amp):
                    logits = self.model(input_ids, pad_mask)  # (batch_size, seq_len, vocab)
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

                total_loss += float(loss.item())

                if step % log_every == 0:
                    avg = total_loss / step
                    print(f"[S3RecPretrain][Epoch {ep}] step {step}/{len(self.train_loader)} | loss={avg:.4f}")

            avg_loss = total_loss / max(1, len(self.train_loader))
            dt = time.time() - t0
            print(f"✅ [S3RecPretrain] Epoch {ep} done | loss={avg_loss:.4f} | {dt:.1f}s")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), best_path)
                print(f"🏆 Best updated! saved: {best_path}")

        print(f"🎯 Best pretrain loss: {best_loss:.6f}")
        return best_path
