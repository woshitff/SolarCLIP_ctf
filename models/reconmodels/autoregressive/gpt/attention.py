from collections import OrderedDict

import torch
import torch.nn as nn

from einops import rearrange


class patch_norm(nn.Module):
    def __init__(self, d_model = 768, norm_type = 'bn1d', eps = 1e-5):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == 'bn1d':
            self.norm = nn.BatchNorm1d(d_model, eps)
        elif norm_type == 'ln':
            self.norm = nn.LayerNorm(d_model, eps)
        else:
            raise ValueError('norm_type should be bn1d or ln')

    def forward(self, x):
        # x.size: (b, num_patches, d_model) (b (n_t n_h n_w) d)
        # x shape: (batch_size, num_frames, num_patches, d_model)
        if self.norm_type == 'bn1d':
            x = rearrange(x, 'b p d -> b d p')
            x = self.norm(x)
            x = rearrange(x, 'b d p -> b p d')
        elif self.norm_type == 'ln':
            x = self.norm(x)
        else:
            raise ValueError('norm_type should be bn1d or ln')
        return x
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, norm_type: str = 'bn1d'):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = patch_norm(d_model = d_model, norm_type = norm_type)
        self.mlp = nn.Sequential(OrderedDict([ 
            ("c_fc", nn.Linear(d_model, d_model * 4)),  
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = patch_norm(d_model = d_model, norm_type = norm_type)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
   def __init__(self, width: int, layers: int, heads: int, drop_out: float=0.0, attn_mask: torch.tensor = None):
      super().__init__()
      self.width = width
      self.layers = layers
      self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
      self.dropout = nn.Dropout(drop_out)

   def forward(self, x: torch.Tensor):
      
      return self.resblocks(x)