import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistVAE(L.LightningModule):
    """
    Variational AutoEncoder for MNIST

    Hydra 설정 사용:
        model:
            input_dim: 784  # 28*28
            hidden_dims: [256, 128]
            latent_dim: 2
        training:
            lr: 1e-3
            weight_decay: 0.1
            kl_weight: 1.0
    """

    def __init__(
        self,
        input_dim=784,
        hidden_dims=[256, 128],
        latent_dim=2,
        lr=1e-3,
        weight_decay=0.1,
        kl_weight=1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.kl_weight = kl_weight

        # Encoder layers 동적 생성
        encoder_layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # 잠재 공간 (mu, logvar)
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder layers 동적 생성 (encoder의 역순)
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        decoder_layers.append(nn.ReLU())

        reversed_dims = hidden_dims[::-1]
        for i in range(len(reversed_dims) - 1):
            decoder_layers.append(nn.Linear(reversed_dims[i], reversed_dims[i + 1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(reversed_dims[-1], input_dim))
        decoder_layers.append(nn.Sigmoid())  # 이미지 픽셀 값 [0, 1] 범위
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """
        인코더: 이미지 → 잠재 공간 (mu, logvar)

        Args:
            x: 입력 이미지 (batch_size, input_dim)

        Returns:
            mu: 잠재 분포의 평균 (batch_size, latent_dim)
            logvar: 잠재 분포의 로그 분산 (batch_size, latent_dim)
        """
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
            std = torch.exp(0.5 * logvar)  # sigma = sqrt(var) 과 동일
            eps = torch.randn_like(std)  # std와 동일 사이즈의 표준정규분포 샘플링
            z = mu + eps * std
        else:
            z = mu  # 추론 시에는 평균값만 사용
        return z

    def decode(self, z):
        """
        디코더: 잠재 벡터 → 재구성된 이미지

        Args:
            z: 잠재 벡터 (batch_size, latent_dim)

        Returns:
            x_hat: 재구성된 이미지 (batch_size, input_dim)
        """
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        """
        전체 순전파: 인코딩 → 재매개변수화 → 디코딩

        Args:
            x: 입력 이미지 (batch_size, input_dim)

        Returns:
            x_hat: 재구성된 이미지
            mu: 잠재 분포의 평균
            logvar: 잠재 분포의 로그 분산
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def training_step(self, batch, batch_idx):
        """
        학습 스텝: Reconstruction Loss + KL Divergence 계산
        """
        x, _ = batch
        x = x.view(x.size(0), -1)  # Flatten (batch_size, 784)

        x_hat, mu, logvar = self(x)

        # Reconstruction Loss (Binary Cross Entropy)
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)

        # KL Divergence: KL(q(z|x) || p(z))
        # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Total Loss
        loss = recon_loss + self.kl_weight * kl_loss

        # 로깅
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kl_loss", kl_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        검증 스텝
        """
        x, _ = batch
        x = x.view(x.size(0), -1)

        x_hat, mu, logvar = self(x)

        # Reconstruction Loss
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)

        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Total Loss
        loss = recon_loss + self.kl_weight * kl_loss

        # 로깅
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kl_loss", kl_loss)

    def test_step(self, batch, batch_idx):
        """
        테스트 스텝
        """
        x, _ = batch
        x = x.view(x.size(0), -1)

        x_hat, mu, logvar = self(x)

        # Reconstruction Loss
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)

        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Total Loss
        loss = recon_loss + self.kl_weight * kl_loss

        # 로깅
        self.log("test_loss", loss)
        self.log("test_recon_loss", recon_loss)
        self.log("test_kl_loss", kl_loss)

    def configure_optimizers(self):
        """
        옵티마이저 설정: AdamW + CosineAnnealingLR
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        return [optimizer], [scheduler]

    def sample(self, num_samples=16):
        """
        잠재 공간에서 샘플링하여 이미지 생성

        Args:
            num_samples: 생성할 샘플 개수

        Returns:
            generated: 생성된 이미지 (num_samples, input_dim)
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            generated = self.decode(z)
        return generated
