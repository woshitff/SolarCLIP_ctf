import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import pytorch_lightning as pl

from models.reconmodels.ldm.util import instantiate_from_config
from models.reconmodels.autoregressive.gpt.attention import Transformer

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class patchify(nn.Module):
    def __init__(self, patch_size, stride):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        assert self.patch_size==self.stride, "Patch size and stride should be equal"

    def forward(self, x):
        tokens = x.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride) # (B, C, H, W) -> (B, C, H/patch_size, patch_size, W/patch_size, patch_size)
        tokens = einops.rearrange(tokens, 'b c h n_h w n_w -> b (c n_h n_w) (h w)') # (B, C*patch_size^2, H/patch_size*W/patch_size)

        return tokens
    
class unpatchify(nn.Module):
    def __init__(self, patch_size, stride):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        assert self.patch_size==self.stride, "Patch size and stride should be equal"

    def forward(self, x):
        tokens = einops.rearrange(x, 'b (c n_h n_w) (h w) -> b c h n_h w n_w', n_h=self.patch_size, n_w=self.patch_size) # (B, C*patch_size^2, H/patch_size*W/patch_size) -> (B, C, H/patch_size, patch_size, W/patch_size, patch_size)
        tokens = tokens.fold(2, self.patch_size, self.stride).fold(3, self.patch_size, self.stride) # (B, C, H/patch_size, patch_size, W/patch_size, patch_size) -> (B, C, H, W)

        return tokens
    

class SolarLatentGPT(pl.LightningModule):
    def __init__(self, 
                 in_channels: int=3, 
                 input_size: int=64 , 
                 patch_size: int=2, 
                 embed_dim: int=768, 
                 width: int=64, 
                 heads: int=12,
                 layers: int=12, 
                 drop_out: float=0.1,
                 attn_mask: bool=False,
                 loss_type: str='l2',
                 hmi_vae_config: dict=None,
                 aia0094_vae_config: dict=None
                ):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.block_size = in_channels * input_size ** 2 // (patch_size ** 2)
        assert self.block_size % 2 == 0, "Block size should be even"
        self.max_new_tokens = self.block_size // 2
        self.embed_dim = embed_dim
        self.width = width
        self.heads = heads
        self.layers = layers    
        self.loss_type = loss_type

        scale = width ** -0.5
        self.pos_embedding = nn.Parameter(scale * torch.randn(2 * (input_size // patch_size) ** 2 , width))
        self.embedding = nn.Linear(patch_size ** 2 , embed_dim)
        self.unembedding = nn.Linear(embed_dim, patch_size ** 2)
        self.transformers = Transformer(width, heads, layers, drop_out, attn_mask=None)

        self.instantiate_vae_model(hmi_vae_config)
        self.instantiate_vae_model(aia0094_vae_config)

    def instantiate_vae_model(self, vae_config):
        model = instantiate_from_config(vae_config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        
        print(f"Loaded model")

    def forward(self, idx):

        b, t, _ = idx.size()
        
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, total block size is only {self.block_size}, which is the sum of hmi and 0094 images' tokens"

        pos_embed = self.pos_embedding[:t, :].unsqueeze(0).expand(b, -1, -1)
        token_embed = self.embedding(idx)
        x = token_embed + pos_embed
        
        for layer in self.transformers:
            x = layer(x)
        
        return x
    
    def generate(self, idx, max_new_tokens):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if idx.size(1) <= self.block_size//2:
            idx = self.embedding(idx)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:, :]

            idx_next = self(idx_cond)
            idx = torch.cat((idx, idx_next), dim=1)

        y_hat = idx[:, self.block_size//2:, ]
        y_hat = self.unembedding(y_hat)
        y_hat = unpatchify(self.patch_size, self.patch_size)(y_hat)

        return y_hat
                
    @torch.no_grad()
    def get_latent(self, x, k):
        if k == 'magnet_image':
            mu, logvar = self.hmi_vae.encode(x)
        elif k == '0094_image':
            mu, logvar = self.aia0094_vae.encode(x)
        else:
            raise NotImplementedError(f"Key {k} not supported")
        
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std

        return z

    @torch.no_grad()
    def get_data(self, batch, batch_idx, k):
        if k == 'magnet_image':
            x = batch[:, 0, :, :, :]
        elif k == '0094_image':
            x = batch[:, 1, :, :, :]
        else:
            raise NotImplementedError(f"Key {k} not supported")
        
        latent = self.get_latent(x, k)
        tokens = patchify(self.patch_size, self.patch_size)(latent)

        return tokens
    
    def loss_function(self, y_hat, y, weights=None):
        if self.loss_type == 'l2':
            loss = F.mse_loss(y_hat, y, reduction='mean')
        elif self.loss_type == 'l1':
            loss = F.l1_loss(y_hat, y, reduction='mean')
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not supported")
        
        return loss
        
    def get_loss(self, y_hat, y, weights=None):
        log_prefix = 'train' if self.training else 'val'
        loss_dict = {}

        loss = self.loss_function(y_hat, y, weights)
        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def shared_step(self, batch, batch_idx):
        x = self.get_data(batch, batch_idx, 'magnet_image') # (B, 3*32*32, 4）
        y = self.get_data(batch, batch_idx, '0094_image')

        y_hat = self.generate(x, self.max_new_tokens)

        loss, loss_dict = self.get_loss(y_hat, y)

        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)     
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)     
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    

