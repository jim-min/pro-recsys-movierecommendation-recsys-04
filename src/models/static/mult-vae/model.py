import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiVAE(nn.Module):
    def __init__(self, num_items, hidden_dim1=600, hidden_dim2=200, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim1),
            nn.Tanh(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh(),
        )

        self.mu = nn.Linear(hidden_dim2, hidden_dim2)
        self.logvar = nn.Linear(hidden_dim2, hidden_dim2)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.Tanh(),
            nn.Linear(hidden_dim1, num_items),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        logits = self.decoder(z)
        return logits, mu, logvar
