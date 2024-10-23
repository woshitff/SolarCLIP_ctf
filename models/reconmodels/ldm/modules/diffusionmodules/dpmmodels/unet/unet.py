import math
from functools import partial

import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

from  .utils import SinusoidalPosEmb, RandomOrLearnedSinusoidalPosEmb
from  .attention import MemAttention, LinearMemAttention
from  .norm import RMSNorm
from  .tau_encoder import TauEncoder
from  models.clipmodels.modules.vit import VisionTransformer, Remove_class_token

# ------------ helpers functions ------------

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# ------------------ Unet blocks ------------------
"""
Unet structure:
    -Downsampling Block/Path:

        ResnetBlock -> ResnetBlock -> Attention -> Downsample
        先提取局部特征, 一次低层次局部特征提取, 一次高层次局部特征提取
        然后利用注意力机制提取全局特征，帮助模型捕捉全局依赖
        最后降低分辨率, 提供更广阔的感受野
        -ResnetBlock: extract the features of the input tensor and add it to the output tensor
        -Downsample: downsample the resolution of input tensor, provide a wider field of view
        -skip connection: store the hidden high resolution states for skip connections

    -Middle block:

        -ResnetBlock: extract the features of the input tensor and add it to the output tensor
        -Attention: apply attention mechanism to the output tensor
        -ResnetBlock: extract the features of the input tensor and add it to the output tensor

    -Upsampling Block:

        ResnetBlock -> ResnetBlock -> Attention -> Upsample

        -ResnetBlock: extract the features of the input tensor and add it to the output tensor
        -Upsample: upsample the resolution of input tensor, 
        -skip connection: store the hidden high resolution states for skip connections
"""

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x) # (B, D, H, W) -> (B, D_out, H, W)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return self.dropout(x)    
    
class ResnetBlock(nn.Module):
    """
    extract the features of the input tensor and add it to the output tensor
    """
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb) # (B, D*4) -> (B, dim_out*2)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1') # (B, dim_out*2) -> (B, dim_out*2, 1, 1)
            scale_shift = time_emb.chunk(2, dim = 1) # (B, dim_out*2, 1, 1) -> (B, dim_out, 1, 1), (B, dim_out, 1, 1)

        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)

# ------------------ Unet ------------------

class Unet(nn.Module):
    def __init__(
        self,
        input_size,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults: tuple=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions
        self.input_size = input_size
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim) # init_dim = 64
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)] # dims = [64, 64, 128, 256, 512]
        in_out = list(zip(dims[:-1], dims[1:])) # in_out = [(64, 64), (64, 128), (128, 256), (256, 512)]

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb, # if not self.random_or_learned_sinusoidal_cond: (B, ) -> (B, D)
            nn.Linear(fourier_dim, time_dim), # (B, D) -> (B, D*4)
            nn.GELU(),
            nn.Linear(time_dim, time_dim) # (B, D*4) -> (B, D*4)
        )

        # condition embeddings
        self.cond_vit = VisionTransformer(
            in_channels=1,
            input_resolution=1024,
            patch_size=64,
            width=768,
            layers=12,
            heads=12,
            output_dim=768,
            token_type='all embedding',
            norm_type='bn1d',
        )
        self.Remove_class_token = Remove_class_token
        self.condition_pre_emb = nn.Sequential(
            self.cond_vit,
            self.Remove_class_token(), #(B, L+1, D) -> (B, L, D)
        )
        self.condition_emb = TauEncoder(channels)

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True) # full_attn = (False, False, False, True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages) # full_attn = (False, False, False, True)
        attn_heads = cast_tuple(attn_heads, num_stages) # attn_heads = (4, 4, 4, 4)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages) # attn_dim_head = (32, 32, 32, 32)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(MemAttention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out) # num_resolutions = 4

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            # ind = 0, 1, 2, 3
            # num_resolutions = 4
            is_last = ind >= (num_resolutions - 1) # is_last = False, False, False, True
            attn_klass = FullAttention if layer_full_attn else LinearMemAttention
            self.downs.append(nn.ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearMemAttention

            self.ups.append(nn.ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim) # self.out_dim = default_out_dim = 3 or 6 if learned_variance

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x_self_cond = self.condition_emb(x_self_cond) # (B, 256, 768) -> (B, 3, 64, 64)
            x = torch.cat((x_self_cond, x), dim = 1) # (B, 3, 64, 64) -> (B, 6, 64, 64)
        x = self.init_conv(x) # (B, 6, 64, 64) -> (B, D, 64, 64)
        r = x.clone() # used for residual connection # (B, D, H, W)

        t = self.time_mlp(time) # (B, ) -> (B, D*4)

        h = [] # to store hidden states for skip connections

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t) 
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)
        # x: (B, D, H, W) -> (B, D, H/2, W/2) for each downsample
        # x: (B, 64, 64, 64) -> (B, 64, 32, 32) -> (B, 128, 16, 16) -> (B, 256, 8, 8) -> (B, 512, 4, 4)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)
        # x: (B, 512, 4, 4) -> (B, 512, 4, 4)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)
        # x: (B, 512, 4, 4) -> (B, 512, 8, 8) -> (B, 256, 16, 16) -> (B, 128, 32, 32) -> (B, 64, 64, 64)
        x = torch.cat((x, r), dim = 1)
        # x: (B, 64, 64, 64) -> (B, 128, 64, 64)
        x = self.final_res_block(x, t)
        # x: (B, 128, 64, 64) -> (B, 64, 64, 64)
        return self.final_conv(x)
        # x: (B, 64, 64, 64) -> (B, 3, 64, 64) or (B, 6, 64, 64) if learned_variance

def get_DiUNet_model_from_args(args):
    """
    create a DiUNet model from arguments
    """
    return Unet(
        input_size=args.input_size, # 64
        dim=args.dim, # 隐藏层通道数
        init_dim=args.init_dim, # 初始隐藏层通道数
        out_dim=args.out_dim, # 输出通道数 None
        dim_mults=args.dim_mults, # [1, 2, 4, 8]
        channels=args.channels, # 3
        self_condition=args.self_condition, # True
        learned_variance=args.learned_variance, # True 
        learned_sinusoidal_cond=args.learned_sinusoidal_cond, # False
        random_fourier_features=args.random_fourier_features, # False
        learned_sinusoidal_dim=args.learned_sinusoidal_dim, # 16
        sinusoidal_pos_emb_theta=args.sinusoidal_pos_emb_theta, # 10000
        dropout=args.dropout, # 0.1
        attn_dim_head=args.attn_dim_head, # 32
        attn_heads=args.attn_heads, # 4
        full_attn=args.full_attn, # None
        flash_attn=args.flash_attn # False
    )

