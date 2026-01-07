import torch
import torch.nn.functional as F

def multivae_loss(recon_x, x, mu, logvar, anneal=1.0):
    """
    Multi-VAE Loss = -LogLikelihood(Multinomial) + Beta * KLD

    Args:
        recon_x (torch.Tensor(Batch, Items)): softmax를 거치지 않은 logits
        x (torch.Tensor(Batch, Items)): 원본 x 
        mu (torch.Tensor(Batch, latent_dim)): mu
        logvar (Batch, latent_dim): logvar
        anneal (float, optional): _description_. Defaults to 1.0.
    """
    
    log_softmax_var = F.log_softmax(recon_x, dim=1)
    
    # 관측된 아이템(x=1)에 대한 log probability를 최대화 하도록 함
    recon_loss = -(log_softmax_var * x).sum(dim=1).mean()
    
    # KL Divergence
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    # Total loss
    return recon_loss + anneal * kld_loss
    
    