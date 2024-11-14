from collections import OrderedDict

import torch
import torch.nn as nn

from einops import rearrange

class LayerNorm(nn.LayerNorm):
   #使用的时候需要指定特征维度大小
   #处理float16数据 
   def forward(self, x: torch.Tensor) -> torch.Tensor:
      orig_tpye = x.dtype
      ret = super().forward(x.type(torch.float32))

      return ret.type(orig_tpye)
   
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
        return x * torch.sigmoid(1.702 * x) #GELU的近似计算，GELU是激活函数，用于神经网络的非线性变换
    
class Encoder(nn.Module):
    def __init__(self, 
                 input_size: int = 1024,
                 embed_dim: int = 768,
                 input_dim: int = 1,
                 patch_size: int = 64,
                 dropout_prob: float = 0.1):
        super().__init__()
        if input_size % patch_size != 0:
            raise ValueError(f"Image size {input_size} must be divisible by patch size {patch_size}, now is not divisible.")
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.ln_pre = patch_norm(d_model = embed_dim, norm_type = 'bn1d')
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x) # (B,C,H,W) -> (B,D,H/patch_size,W/patch_size)
        x = x.reshape(x.shape[0], x.shape[1], -1) # (B, D, H/patch_size, W/patch_size) -> (B, D, L=H/patch_size*W/patch_size)
        x = x.permute(0, 2, 1) # (B, D, L) -> (B, L, D)
        x = self.ln_pre(x) # (B, L, D) -> (B, L, D)
        x = self.dropout(x) # (B, L, D) -> (B, L, D)
        return x
    
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

class VisionTransformer(nn.Module):
    def __init__(self, in_channels, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                token_type: str , norm_type: str = 'bn1d'):
        super().__init__()
        self.transformer_token_type = token_type
        self.conv1 = Encoder(input_dim=in_channels, embed_dim=width, input_size=input_resolution, patch_size=patch_size, dropout_prob=0.1)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = patch_norm(width, norm_type)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = patch_norm(width, norm_type)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        #! donot permute here, we focus on the global feature
        # x = x.permute(1, 0, 2)  # BLD -> LBD
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LBD -> BLD

        if self.transformer_token_type == 'class embedding':
            x = self.ln_post(x[:, 0,:]) 
            # return [N, align_dim]
        elif self.transformer_token_type == 'all embedding':
            x = self.ln_post(x)
            # return [N, L+1, align_dim]

        if self.proj is not None:
            x = x @ self.proj   

        return x

    def get_last_selfattention(self, x):
        x = self.conv1(x)
        
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = self.transformer(x)

        return x
 
class BaseVisionTransformer(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 width: int, layers: int, heads: int, 
                 output_dim: int,
                 token_type: str, norm_type: str = 'bn1d'):
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.layers = layers
        self.heads = heads
        self.output_dim = output_dim
        self.token_type = token_type
        self.norm_type = norm_type

        self.proj_in = nn.Parameter(torch.zeros(input_dim, width))
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.positional_embedding = nn.Parameter(torch.zeros((1, width)))
        self.ln_pre = patch_norm(width, norm_type)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = patch_norm(width, norm_type)
        self.proj_out = nn.Parameter(torch.zeros(width, output_dim))

    def forward(self, x: torch.Tensor):
        """
        x: [N, L, C]
        output: [N, L+1, output_dim] if token_type == 'all embedding' else [N, output_dim]
        """
        x = x @ self.proj_in
        
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)
        x = self.transformer(x)

        if self.token_type == 'class embedding':
            x = self.ln_post(x[:, 0,:]) 
            # return [N, align_dim]
        elif self.token_type == 'all embedding':
            x = self.ln_post(x)
            # return [N, L+1, align_dim]

        if self.proj is not None:
            x = x @ self.proj_out   

        return x


class Remove_class_token(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x[:, 1:, :]