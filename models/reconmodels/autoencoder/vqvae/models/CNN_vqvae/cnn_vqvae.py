import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.reconmodels.autoencoder.vqvae.modules.vector_quantize_pytorch import VectorQuantize

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
    
class CNN_VQVAE(nn.Module):
    def __init__(self, 
                 input_size: int = 1024, 
                 image_channels: int = 1, 
                 hidden_dim: int = 64,
                 layers: int = 3,
                 kernel_sizes: list = [7,7,7],
                 strides: list = [4,4,4],
                 group_nums: int = 16,
                 latent_dim: int = 768,
                 codebook_size: int = 512,
                 vq_decay: float = 0.99,
                 vq_commitment_weight: float = 1.0,
                 learnable_codebook: bool = False,
                 ema_update: bool = True,
                 loss_type: str = 'MSE',
                 lambda_vq: float = 1.0):
        super().__init__()
        self.input_size = input_size
        self.image_channels = image_channels
        self.group_nums = group_nums
        self.hidden_dim = hidden_dim
        self.layers = layers  
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.vq_decay = vq_decay
        self.vq_commitment_weight = vq_commitment_weight
        self.learnable_codebook = learnable_codebook
        self.ema_update = ema_update
        self.loss_type = loss_type
        self.lambda_vq = lambda_vq
        self.nh = input_size // np.prod(strides)

        self.encoder_list = nn.ModuleList([
            nn.Conv2d(image_channels, hidden_dim, kernel_size=3, stride=1, padding=1),  # B, 1, 1024, 1024 -> B, 128, 1024, 1024
            nn.ELU()
        ])
        for i, kernel_size, stride in zip(range(self.layers), self.kernel_sizes, self.strides):
            self.encoder_list.append(VAE_ResidualBlock(hidden_dim*(2**i), hidden_dim*(2**(i+1)), kernel_size=kernel_size, stride=stride, padding=kernel_size//2))
        self.encoder_list.extend([
            nn.GroupNorm(group_nums, hidden_dim*(2**self.layers)), # B, 1024, 16, 16 -> B, 1024, 16, 16
            nn.ELU(),
            nn.Conv2d(hidden_dim*(2**self.layers), self.latent_dim, kernel_size=1, stride=1, padding=0), # B, 1024, 16, 16 -> B, 6, 16, 16
            nn.BatchNorm2d(self.latent_dim)
        ])
        self.encoder = nn.Sequential(*self.encoder_list)

        self.vq = VectorQuantize(
            dim=self.latent_dim,
            codebook_size=self.codebook_size,
            decay=self.vq_decay,
            commitment_weight=self.vq_commitment_weight,
            accept_image_fmap=True,
            learnable_codebook=self.learnable_codebook,
            ema_update=self.ema_update
        )

        self.decoder_list = nn.ModuleList([
            nn.ConvTranspose2d(self.latent_dim, hidden_dim*(2**self.layers), kernel_size=3, stride=1, padding=1), # B, 3, 16, 16 -> B, 1024, 16, 16
            nn.ELU()
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
    
    def forward(self, x):
        z = self.encoder(x)
        quantized, indices, commit_loss = self.vq(z)
        recon_x = self.decoder(quantized)
        return recon_x, indices, commit_loss
    
    def loss_function(self, recon_x, x, weights, commit_loss, lambda_vq):
        if self.loss_type == 'MSE':
            with torch.no_grad():
                RECON_LOSS = F.mse_loss(recon_x, x, reduction='mean')
            RECON_LOSS_weighted = weights*F.mse_loss(recon_x, x, reduction='none')
            RECON_LOSS_weighted = RECON_LOSS_weighted.mean()
        elif self.loss_type == 'BCE':
            RECON_LOSS = F.binary_cross_entropy(recon_x, x, reduction='sum')
        else:
            raise ValueError(f"loss_type {self.loss_type} is not supported")
        
        commit_loss = commit_loss.mean()
        total_loss = RECON_LOSS_weighted + lambda_vq*commit_loss

        return total_loss, RECON_LOSS.detach(), RECON_LOSS_weighted.detach(), commit_loss.detach()
    
    def sample(self, x):
        z = self.encoder(x)
        quantized, _ , _= self.vq(z)
        return quantized
    
    def generate(self, quantized):
        return self.decoder(quantized)
