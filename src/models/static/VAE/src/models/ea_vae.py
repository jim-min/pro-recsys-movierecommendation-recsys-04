import torch
import torch.nn as nn
import torch.nn.functional as F

class EAVAE(nn.Module):
    def __init__(self: nn.Module, ease_weight: torch.Tensor, q_dims, p_dims=None, ease_rate=0.5, dropout_rate=0.5):
        """
        Args:
            ease_weight (torch.Tensor): EASE모델의 weight
            p_dims (list): 레이어 차원 리스트 [input_dim, hidden_dim, ... , latent_dim]
            dropout_rate (float, optional): Defaults to 0.5.
        """
        
        super(EAVAE, self).__init__()
        self.q_dims = q_dims
        self.p_dims = q_dims[::-1]
        if p_dims:
            self.p_dims = p_dims
        
        # EASE의 가중치를 참조용으로 따로 저장
        # 학습이 되지않는 텐서    
        self.register_buffer('ease_weight', ease_weight.clone().detach())
        self.ease_rate = ease_rate
        self.ease_ln = nn.LayerNorm(self.q_dims[0], elementwise_affine=False)
        self.gate_layer = nn.Sequential(
            nn.Linear(q_dims[0], q_dims[0]),
            nn.Sigmoid()
        )
        
        # Encoder (q_z|x)
        # input -> hidden -> latent
        encoder_modules = []
        
        for i in range(len(self.q_dims)-2):
            encoder_modules.append(nn.Linear(self.q_dims[i], self.q_dims[i+1]))
            encoder_modules.append(nn.Tanh())
        encoder_modules.append(nn.Linear(self.q_dims[-2], self.q_dims[-1]*2))

        self.encoder = nn.Sequential(*encoder_modules)
        
        
        # Decoder (p_x|z)
        # latent -> hidden -> input
        decoder_modules = []
        for i in range(len(self.p_dims)-2):
            decoder_modules.append(nn.Linear(self.p_dims[i], self.p_dims[i+1]))
            decoder_modules.append(nn.Tanh())
        decoder_modules.append(nn.Linear(self.p_dims[-2], self.p_dims[-1]))
        
        self.decoder = nn.Sequential(*decoder_modules)
            
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 가중치 초기화
        self.apply(self.init_weights)
        
        self.output_alpha = nn.Parameter(torch.tensor(20.0))
        self.output_beta = nn.Parameter(torch.tensor(0.0))
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Linear 모듈 xavier 초기화(tanh 활성함수 사용)
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                # bias의 경우 0으로 초기화
                nn.init.constant_(m.bias, 0.0)
        
    # Encode
    def encode(self, x):
        # (Batch, latent_dim * 2)
        h = self.encoder(x)
        
        # 출력을 반으로 쪼개서 mu와 logvar로 사용
        mu = h[:, :self.q_dims[-1]]
        logvar = h[:, self.q_dims[-1]:]
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # 학습의 경우만 sampling
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        # test의 경우 mu를 사용
        else:
            return mu
    
    # Decode
    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat
    
    def forward(self, x):
        # L2 Normalization
        
        # ease_logits = x @ self.ease_weight
        # ease_norm = self.ease_ln(ease_logits)
        # ease_prob = torch.sigmoid(ease_norm)
        # row vector마다 기존의 multi-hot vector를 크기를 1로 정규화
        
        # alpha = self.gate_layer(x)
        
        # enhanced_input = x + alpha * ease_prob
        
        x_norm = F.normalize(x, p=2, dim=1)
        
        # Dropout
        x_drop = self.dropout(x_norm)
        
        # Encoding
        mu, logvar = self.encode(x_drop)
        
        # samplings
        z = self.reparameterize(mu, logvar)
        
        # Decoding
        vae_logits = self.decode(z)
        
        ease_base = x @ self.ease_weight
        ease_norm = F.normalize(ease_base, p=2, dim=1)
        
        # ease_scale_factor = 100.0  # 50~100 사이 값 추천
        # ease_base_scaled = ease_base / ease_scale_factor
        
        final_logits = self.output_alpha * ease_norm + self.output_beta * vae_logits
        
        return final_logits, mu, logvar