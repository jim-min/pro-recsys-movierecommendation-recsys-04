import logging
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class MultiVAE(L.LightningModule):
    """
    Variational AutoEncoder for Collaborative Filtering

    Hydra 설정 사용:
        model:
            hidden_dims: [600, 200]
            dropout: 0.5
        training:
            lr: 1e-3
            weight_decay: 0.01
            kl_max_weight: 0.2
            kl_anneal_steps: 20000
    """

    def __init__(
        self,
        num_items,
        hidden_dims=[600, 200],  # 임의의 배열
        dropout=0.5,
        lr=1e-3,
        weight_decay=0.01,
        kl_max_weight=0.2,
        kl_anneal_steps=20000,
    ):
        super().__init__()
        self.save_hyperparameters()  # 자동 저장 (체크포인트에 포함)

        self.num_items = num_items
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.kl_max_weight = kl_max_weight
        self.kl_anneal_steps = kl_anneal_steps

        # Encoder layers 동적 생성
        encoder_layers = []
        input_dim = num_items
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # 잠재 공간 (mu, logvar), 마지막 입력 차원
        latent_dim = hidden_dims[-1]
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

        # Decoder layers 동적 생성 (encoder의 역순)
        decoder_layers = []
        reversed_dims = hidden_dims[::-1]
        for i in range(len(reversed_dims) - 1):
            decoder_layers.append(nn.Linear(reversed_dims[i], reversed_dims[i + 1]))
            decoder_layers.append(nn.Tanh())
        decoder_layers.append(nn.Linear(reversed_dims[-1], num_items))
        self.decoder = nn.Sequential(*decoder_layers)

        self._init_weights()

        # 모델 초기화 로그
        log.info(f"MultiVAE initialized with {num_items} items")
        log.info(f"Hidden dimensions: {hidden_dims}")
        log.info(f"Dropout: {dropout}, LR: {lr}, Weight decay: {weight_decay}")
        log.info(f"KL max weight: {kl_max_weight}, KL anneal steps: {kl_anneal_steps}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        """
        인코더: 유저-아이템 벡터 → 잠재 공간 (mu, logvar)

        Args:
            x: 유저-아이템 상호작용 벡터 (batch_size, num_items)

        Returns:
            latent_dim 차원의 다변량 가우시안 분포를 리턴(각 Row_user 별로 latent_dim 차원의 mu 벡터와 cov(실재는 logvar) )
            mu: 잠재 분포의 평균 (batch_size, latent_dim),
            logvar: 잠재 분포의 로그 분산 (batch_size, latent_dim)
        """
        # Dropout 먼저 적용 (정규화 전)
        x = F.dropout(x, self.dropout, training=self.training)
        # L2 정규화 (dropout 후)
        x = F.normalize(x, p=2, dim=1)

        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        재매개변수화 기법: z = mu + eps * std

        Args:
            mu: 잠재 분포의 평균
            logvar: 잠재 분포의 로그 분산

        Returns:
            z: 샘플링된 잠재 벡터
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # 추론 시에는 평균값만 사용

        return z

    def decode(self, z):
        """
        디코더: 잠재 벡터 → 아이템 점수

        Args:
            z: 잠재 벡터 (batch_size, latent_dim)

        Returns:
            logits: 재구성된 아이템 점수 (batch_size, num_items)
        """
        logits = self.decoder(z)
        return logits

    def forward(self, x):
        """
        전체 순전파: 인코딩 → 재매개변수화 → 디코딩

        Args:
            x: 유저-아이템 상호작용 벡터 (batch_size, num_items)

        Returns:
            logits: 재구성된 아이템 점수 (batch_size, num_items)
            mu: 잠재 분포의 평균
            logvar: 잠재 분포의 로그 분산
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)

        return logits, mu, logvar

    def _kl_weight(self):
        """KL Annealing: 학습 초기엔 reconstruction에 집중, 점진적으로 KL weight 증가"""
        return min(
            self.kl_max_weight,
            self.kl_max_weight * self.global_step / self.kl_anneal_steps,
        )

    def training_step(self, batch, batch_idx):
        """
        학습 스텝: Reconstruction Loss + KL Loss 계산 및 KL Annealing 적용
        """
        x = (
            batch[0] if isinstance(batch, (tuple, list)) else batch
        )  # 유저-아이템 상호작용 벡터
        logits, mu, logvar = self(x)

        # Reconstruction Los
        log_softmax = F.log_softmax(logits, dim=1)
        recon_loss = -(log_softmax * x).sum(dim=1).mean()

        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # Total Loss with KL Annealing
        loss = recon_loss + self._kl_weight() * kl_loss

        # 로깅 (TensorBoard)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kl_loss", kl_loss)
        self.log("kl_weight", self._kl_weight())

        return loss

    def validation_step(self, batch, batch_idx):
        """
        검증 스텝: KL weight는 최대값 사용 (annealing 없음)
        """
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        logits, mu, logvar = self(x)

        # Reconstruction Loss
        log_softmax = F.log_softmax(logits, dim=1)
        recon_loss = -(log_softmax * x).sum(dim=1).mean()

        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # Total Loss
        loss = recon_loss + self.kl_max_weight * kl_loss

        # 로깅 (TensorBoard)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kl_loss", kl_loss)

    def configure_optimizers(self):
        """
        옵티마이저 설정: Adam + ReduceLROnPlateau 스케줄러
        val_loss 기준으로 자동 learning rate 조정
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=15,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
