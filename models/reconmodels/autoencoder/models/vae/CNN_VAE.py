import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models.reconmodels.autoencoder.util import config_optimizers
from models.reconmodels.autoencoder.util import instantiate_from_config
"""
SolarReconModel_VAE Model.
This model use VAE architecture to reconstruct the solar image without clip model.
"""

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias = True, out_proj_bias= True):
        super().__init__()
        
        #Combining the Wq, Wk, and Wv matrices into one
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias= in_proj_bias)
        
        #Represent the Wo Matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias= out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x, causal_mask = False):
        
        input_shape = x.shape
        
        batch_size, sequence_length, d_embed = input_shape
        
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k ,v = self.in_proj(x).chunk(3, dim= -1)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)
        
        weight = q @ k.transpose(-1,-2)
        
        if causal_mask:     
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim = -1)
        
        output = weight @ v
        
        output = output.transpose(1,2)
        
        output = output.reshape(input_shape)
        
        output = self.out_proj(output)
        
        return output

class VAE_AttentionBlock(nn.Module):
    def __init__(self, 
                 channels, 
                 num_groups: int = 32):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups, channels)
        self.attention = SelfAttention(1,channels)
        
    def forward(self,x):
        
        residue = x
        x = self.groupnorm(x)
        
        n, c, h, w = x.shape
        x = x.view(n ,c,h *w)
        x = x.permute(0,2,1)                
        x = self.attention(x)
        x = x.permute(0,2,1)        
        x = x.view(n,c,h,w)
    
        x += residue
        return x
    
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups: int = 16,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.groupnorm_pre = nn.GroupNorm(num_groups, out_channels//2)
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.groupnorm_post = nn.GroupNorm(num_groups, out_channels)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.nonlinear = nn.ELU()
            
    def forward(self,x):
        residue = self.residual_layer(x)
        x = self.nonlinear(self.groupnorm_pre(self.conv1(x)))
        x = self.conv2(x)
        return self.nonlinear(self.groupnorm_post(x + residue))
    
    
class CNN_VAE(pl.LightningModule):
    def __init__(self,
                 ckpt_path: str = None,
                 vae_modal: str = 'magnet_image',
                 input_size: int = 1024,
                 image_channels: int = 1,
                 hidden_dim: int = 64,
                 layers: int = 3,
                 kernel_sizes: list = [7, 7, 7],
                 strides: list = [4, 4, 4],
                 group_nums: int = 16,
                 latent_dim: int = 3,
                 loss_type: str = 'MSE',
                 lambda_kl: float = 1.0,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.vae_modal = vae_modal
        self.input_size = input_size
        self.image_channels = image_channels
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.group_nums = group_nums
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        self.lambda_kl = lambda_kl

        self.encoder_list = nn.ModuleList([
            nn.Conv2d(image_channels, hidden_dim, kernel_size=3, stride=1, padding=1),  # B, 1, 1024, 1024 -> B, 128, 1024, 1024
            nn.ELU()
        ])
        for i, kernel_size, stride in zip(range(self.layers), self.kernel_sizes, self.strides):
            self.encoder_list.append(VAE_ResidualBlock(hidden_dim*(2**i), hidden_dim*(2**(i+1)), kernel_size=kernel_size, stride=stride, padding=kernel_size//2))
        self.encoder_list.extend([
            VAE_ResidualBlock(hidden_dim*(2**self.layers), hidden_dim*(2**(self.layers)), kernel_size=3, stride=1, padding=1),  # B, 1024, 16, 16 -> B, 1024, 16, 16
            VAE_ResidualBlock(hidden_dim*(2**(self.layers)), hidden_dim*(2**(self.layers)), kernel_size=3, stride=1, padding=1),  # B, 1024, 16, 16 -> B, 1024, 16, 16
            nn.GroupNorm(group_nums, hidden_dim*(2**self.layers)), # B, 1024, 16, 16 -> B, 1024, 16, 16
            nn.ELU(),   
            nn.Conv2d(hidden_dim*(2**self.layers), self.latent_dim*2, kernel_size=1, stride=1, padding=0), # B, 1024, 16, 16 -> B, 6, 16, 16
        ])
        self.encoder = nn.Sequential(*self.encoder_list)

        self.decoder_list = nn.ModuleList([
            nn.ConvTranspose2d(self.latent_dim, hidden_dim*(2**self.layers), kernel_size=3, stride=1, padding=1), # B, 3, 16, 16 -> B, 1024, 16, 16
            nn.ELU(),
            nn.GroupNorm(group_nums, hidden_dim*(2**self.layers)), # B, 1024, 16, 16 -> B, 1024, 16, 16
            VAE_ResidualBlock(hidden_dim*(2**(self.layers)), hidden_dim*(2**(self.layers)), kernel_size=3, stride=1, padding=1),  # B, 1024, 16, 16 -> B, 1024, 16, 16
            VAE_ResidualBlock(hidden_dim*(2**(self.layers)), hidden_dim*(2**(self.layers)), kernel_size=3, stride=1, padding=1),  # B, 1024, 16, 16 -> B, 1024, 16, 16
        ])
        for i, kernel_size, stride in zip(range(self.layers-1, -1, -1), self.kernel_sizes[::-1], self.strides[::-1]):
            self.decoder_list.extend([
                nn.Upsample(scale_factor=stride, mode='nearest'), # B, 1024, 16, 16 -> B, 1024, 64, 64
                VAE_ResidualBlock(hidden_dim*(2**(i+1)), hidden_dim*(2**i), kernel_size=kernel_size, stride=1, padding=kernel_size//2) # B, 1024, 64, 64 -> B, 512, 64, 64
            ])
        self.decoder_list.extend([
            nn.GroupNorm(group_nums, hidden_dim), # B, 512, 64, 64 -> B, 512, 64, 64
            nn.ELU(),
            nn.Conv2d(hidden_dim, image_channels, kernel_size=3, stride=1, padding=1), # B, 512, 64, 64 -> B, 1, 64, 64
        ])
        self.decoder = nn.Sequential(*self.decoder_list)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.load_state_dict(checkpoint['model'])

        print(f"Loaded model from {ckpt_path}")

    def encode(self, x):
        """
        x: (B, C, H, W) eg: (B, 1, 1024, 1024)
        output: (B, C_out, H_out, W_out) eg: (B, 3, 64, 64)
        """
        x = self.encoder(x) # (B, C, H, W) -> (B, C_out, H_out, W_out)
        mu, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30, 30)
        return mu, logvar

    def reparameterize(self, mu, logvar, scale=1.0):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        if self.training:   
            z = mu + eps*std
        else:
            z = mu
        z = z*scale
        return z

    def decode(self, z):
        """
        z: (B, latent_dim, H_out, W_out) eg: (B, 3, 16, 16)
        output: (B, input_dim, H, W) eg: (B, 1, 1024, 1024)
        """
        return self.decoder(z)

    def sample(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return self.decode(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = mu
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def get_input(self, batch, k):
        if k == 'magnet_image':
            x = batch[:, 0, :, :, :]
        elif k == '0094_image':
            x = batch[:, 1, :, :, :]
        else:
            raise NotImplementedError(f"Key {k} not supported")
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def loss_function(self, recon_x, x, weights, mu, logvar, lambda_kl):
        if self.loss_type == 'MSE':
            with torch.no_grad():
                RECON_LOSS = F.mse_loss(recon_x, x, reduction='mean')
            RECON_LOSS_weighted = weights*F.mse_loss(recon_x, x, reduction='none')
            RECON_LOSS_weighted = RECON_LOSS_weighted.mean()
        elif self.loss_type == 'BCE':
            RECON_LOSS = F.binary_cross_entropy(recon_x, x, reduction='sum')
        else:
            raise ValueError(f"loss_type {self.loss_type} is not supported")
        KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))/mu.numel()

        total_loss = RECON_LOSS_weighted + KLD*lambda_kl
        return total_loss, RECON_LOSS.detach(), RECON_LOSS_weighted.detach(), KLD.detach()
    
    def get_loss(self, recon_x, x, weights, mu, logvar, lambda_kl):
        log_prefix = 'train' if self.training else 'val'
        loss_dict = {}

        loss, recon_loss, recon_loss_weighted, kl_loss = self.loss_function(recon_x, x, weights, mu, logvar, lambda_kl)
        loss_dict.update({f'{log_prefix}/loss': loss})
        loss_dict.update({f'{log_prefix}/recon_loss': recon_loss})
        loss_dict.update({f'{log_prefix}/recon_loss_weighted': recon_loss_weighted})
        loss_dict.update({f'{log_prefix}/kl_loss': kl_loss})

        return loss, loss_dict
    
    def shared_step(self, batch, batch_idx):
        x = self.get_input(batch, self.vae_modal)
        recon_x, mu, logvar = self(x)
        weights = torch.ones_like(x)
        loss, loss_dict = self.get_loss(recon_x, x, weights, mu, logvar, self.lambda_kl)
        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_recon', loss_dict['train/recon_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_recon_weighted', loss_dict['train/recon_loss_weighted'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_kl', loss_dict['train/kl_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_recon', loss_dict['val/recon_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_recon_weighted', loss_dict['val/recon_loss_weighted'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_kl', loss_dict['val/kl_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt, scheduler = config_optimizers(self.learning_optimizer, self.parameters(), lr, self.learning_schedule)
        return (opt, scheduler)

    def log_images(self, batch, N=2):
        log = dict()

        x = self.get_input(batch, self.vae_modal)
        N = min(N, x.shape[0])
        log['inputs'] = x[:N]

        self.eval()
        with torch.no_grad():
            recon_x, mu, logvar = self(x) 
            samples = self.sample(x)
        log['recon'] = recon_x[:N]
        log['mu'] = mu[:N]
        log['samples'] = samples[:N]
        self.train()

        return log
    

class aia0094_CNN_VAE(CNN_VAE):
    def __init__(self, loss_config, **kwargs):
        super().__init__(**kwargs)

        self.loss = instantiate_from_config(loss_config)

    def shared_step(self, batch, batch_idx):
        x = self.get_input(batch, '0094_image')
        recon_x, mu, logvar = self(x)
        weights = torch.ones_like(x)
        loss, dict_loss = self.loss(recon_x, x, weights, mu, logvar, self.lambda_kl)
        return loss, dict_loss


# version 0-- : change the loss function 
class aia0094_CNN_VAE_v01(CNN_VAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def loss_function(self, recon_x, x, weights, mu, logvar, lambda_kl):
        if self.loss_type == 'MSE':
            with torch.no_grad():
                RECON_LOSS = F.mse_loss(recon_x, x, reduction='mean')
            RECON_LOSS_weighted = weights*F.mse_loss(recon_x, x, reduction='none')
            RECON_LOSS_weighted = RECON_LOSS_weighted.mean()
        elif self.loss_type == 'BCE':
            RECON_LOSS = F.binary_cross_entropy(recon_x, x, reduction='sum')
        elif self.loss_type == 'L1 + MSE':
            with torch.no_grad():
                RECON_LOSS = 0.5 * F.l1_loss(recon_x, x, reduction='mean') + 0.5 *F.mse_loss(recon_x, x, reduction='mean')
            RECON_LOSS_weighted = weights*0.5 * F.l1_loss(recon_x, x, reduction='none') + 0.5 *F.mse_loss(recon_x, x, reduction='none')
            RECON_LOSS_weighted = RECON_LOSS_weighted.mean()
        else:
            raise ValueError(f"loss_type {self.loss_type} is not supported")
        KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))/mu.numel()

        total_loss = RECON_LOSS_weighted + KLD*lambda_kl
        return total_loss, RECON_LOSS.detach(), RECON_LOSS_weighted.detach(), KLD.detach()

class aia0094_CNN_VAE_v02(aia0094_CNN_VAE_v01):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def gradient_loss(recon_x, x):
        grad_recon_x_h = torch.abs(recon_x[:, :, :, :-1] - recon_x[:, :, :, 1:])
        grad_x_h = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        
        grad_recon_x_v = torch.abs(recon_x[:, :, :-1, :] - recon_x[:, :, 1:, :])
        grad_x_v = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        
        loss_h = F.l1_loss(grad_recon_x_h, grad_x_h)
        loss_v = F.l1_loss(grad_recon_x_v, grad_x_v)
        
        return loss_h + loss_v
    
    def loss_function(self, recon_x, x, weights, mu, logvar, lambda_kl):
        if self.loss_type == 'MSE':
            with torch.no_grad():
                RECON_LOSS = F.mse_loss(recon_x, x, reduction='mean')
            RECON_LOSS_weighted = weights*F.mse_loss(recon_x, x, reduction='none')
            RECON_LOSS_weighted = RECON_LOSS_weighted.mean()
        elif self.loss_type == 'BCE':
            RECON_LOSS = F.binary_cross_entropy(recon_x, x, reduction='sum')
        elif self.loss_type == 'L1 + MSE':
            with torch.no_grad():
                RECON_LOSS = 0.5 * F.l1_loss(recon_x, x, reduction='mean') + 0.5 *F.mse_loss(recon_x, x, reduction='mean')
            RECON_LOSS_weighted = weights*0.5 * F.l1_loss(recon_x, x, reduction='none') + 0.5 *F.mse_loss(recon_x, x, reduction='none')
            RECON_LOSS_weighted = RECON_LOSS_weighted.mean()
        else:
            raise ValueError(f"loss_type {self.loss_type} is not supported")
        KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))/mu.numel()
        GRAD_LOSS = self.gradient_loss(recon_x, x)/mu.numel()
        total_loss = RECON_LOSS_weighted + KLD*lambda_kl + GRAD_LOSS*0.1
        return total_loss, RECON_LOSS.detach(), RECON_LOSS_weighted.detach(), KLD.detach()