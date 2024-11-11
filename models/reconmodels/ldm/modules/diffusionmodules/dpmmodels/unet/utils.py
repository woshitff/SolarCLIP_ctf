import math

import torch
import torch.nn as nn

from einops import rearrange

def divisible_by(numer, denom):
    return (numer % denom) == 0

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim # dim = 64
        self.theta = theta

    def forward(self, x):  # x.shape = (B, )
        device = x.device
        half_dim = self.dim // 2 
        emb = math.log(self.theta) / (half_dim - 1) 
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] 
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb 
    # shape = (B, dim)

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2), 'dimension must be divisible by 2'
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered