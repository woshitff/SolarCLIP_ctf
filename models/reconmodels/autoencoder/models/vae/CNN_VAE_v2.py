import os

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models.reconmodels.autoencoder.util import config_optimizers
from models.reconmodels.autoencoder.util import instantiate_from_config
"""
SolarReconModel_VAE_v2 Model.
This model use VAE architecture like autoencoderKL in LDM to reconstruct the solar image without clip model.
see https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
"""
    
def Normalize(in_channels, num_groups=8):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class SelfAttention(nn.Module):
    def __init__(self, dim):
        
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = self.softmax(q @ k.transpose(-2, -1) / (x.shape[-1] ** 0.5))
        return attn_weights @ v
class FFN(nn.Module):
    def __init__(self, dim=32*32*32):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(dim, dim*2)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim*2, dim*2)
        self.linear3 = nn.Linear(dim*2, dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten the input
        x1 = self.linear1(x)
        x1 = self.activation(x1)
        x1 = self.linear2(x1)
        x1 = self.activation(x1)
        x1 = self.linear3(x1)
        x1 = self.activation(x1)
        x1 = x1.view(x1.shape[0], 32, 32, 32)
        x = x.view(x.shape[0], 32, 32, 32)
        return x + x1
    
class transformNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = SelfAttention(dim)
        self.norm = nn.LayerNorm(dim)  # 归一化操作
        #零初始化
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):

        batch_size, channels, h, w = x.shape  # 32
        
        # Reshape: (B, 32, 32, 32) -> (B, 32, 1024)
        x = x.view(batch_size, channels, -1)
        
        # Self-Attention: (B, 32, 1024) -> (B, 32, 1024)
        x = self.attn(x) + x

        # Norm: 归一化 (B, 32, 1024)
        x = self.norm(x)
        
        # Reshape: (B, 32, 1024) -> (B, 32, 32, 32)
        x = x.view(batch_size, channels, h, w)
        
        return x
    
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, 
                 in_channels, out_channels=None, 
                 conv_shortcut=False,dropout,):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.nonlinear = nonlinearity
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.conv1(self.nonlinear(self.norm1(h)))
        h = self.conv2(self.dropout(self.nonlinear(self.norm2(h))))

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        # return LinAttnBlock(in_channels)
        raise NotImplementedError('linear is not support now')

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)
        # self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        # h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        # curr_res = resolution // 2**(self.num_resolutions-1)
        curr_res = 1
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)
        # self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        # h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
    
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class CNN_VAE(pl.LightningModule):
    def __init__(self,
                 ckpt_path: str = None,
                 vae_modal: str = 'magnet_image',
                 kl_weight: float = 1.0,
                 loss_type: str = 'MSE',
                 dd_config: dict = None,
                 if_classify: bool = False,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.vae_modal = vae_modal
        self.lambda_kl = kl_weight
        self.loss_type = loss_type

        self.encoder = Encoder(**dd_config)
        self.decoder = Decoder(**dd_config)
        self.feature_spatial = dd_config['resolution'] // math.prod(dd_config['ch_mult'])
        self.feature_dim = dd_config['z_channels']
        self.if_classify = if_classify
        if if_classify:
            self.class_block = nn.Sequential(
                AttentionPool2d(
                    spacial_dim=self.feature_spatial,
                    embed_dim=self.feature_dim,
                    num_heads=4,
                    output_dim=512
                ),
                    nn.LayerNorm(512),
            )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, ckpt_path):
        if os.path.splitext(ckpt_path)[-1] == '.pt':
            checkpoint = torch.load(ckpt_path)
            self.load_state_dict(checkpoint['model'], strict=False)
        elif os.path.splitext(ckpt_path)[-1] == '.ckpt':
            checkpoint = torch.load(ckpt_path)
            if 'loss' in checkpoint['state_dict']:
                del checkpoint['state_dict']['loss']
            self.load_state_dict(checkpoint['state_dict'], strict=False)

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

    def get_logit(self, mu):
        """
        mu: (B, C_out, H_out, W_out) eg: (B, 3, 64, 64)
        logit: (B, 512)
        """
        assert self.if_classify, "if_classify must be True"
        logit = self.class_block(mu)
        return logit

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
        # z = self.reparameterize(mu, logvar)
        z = mu
        # print(z.shape)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def get_input(self, batch, k):
        # if k == 'hmi_image':
        #     x = batch[:, 0, :, :, :]
        # elif k == 'aia0094_image':
        #     # x = batch[:, 1, :, :, :]
        #     x = batch[:, 0, :, :, :]
            
        # else:
        #     x = batch[k]
        if k is not None:
            x = batch[k]
        else:
            raise NotImplementedError(f"Key {k} not supported")
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def calculate_loss(self, x, return_moment=False):
        """
        a method to get the loss only from input x
        """
        recon_x, mu, logvar = self(x)
        if self.loss_type == 'MSE':
            RECON_LOSS = F.mse_loss(recon_x, x, reduction='mean')
        elif self.loss_type == 'BCE':
            RECON_LOSS = F.binary_cross_entropy(recon_x, x, reduction='sum')
        else:
            raise ValueError(f"loss_type {self.loss_type} is not supported")
        KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))/mu.numel()

        if return_moment:
            return RECON_LOSS, KLD, mu, logvar, recon_x
        else:
            return RECON_LOSS, KLD
    
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
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt, scheduler = config_optimizers(self.learning_optimizer, self.parameters(), lr, self.learning_schedule)
        return (opt, scheduler)

    def log_images(self, batch, N=2):
        print('Begin to log images')
        log = dict()
        modals = dict()

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
        modals['inputs'] = self.vae_modal
        modals['recon'] = self.vae_modal
        modals['mu'] = self.vae_modal
        modals['samples'] = self.vae_modal

        print('Log images down')
        return log, modals
    
class CNN_VAE_two(pl.LightningModule):
    def __init__(self,
                 ckpt_path: str = None,
                 vae_modal: str = 'magnet_image',
                 key: str = 'hmi',
                 kl_weight: float = 1.0,
                 loss_type: str = 'MSE',
                 dd_config: dict = None,
                 first_stage_config: dict = None,
                 train_first_stage: bool = False,
                 norm:bool  = False,
                 if_classify: bool = False,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.vae_modal = vae_modal
        self.lambda_kl = kl_weight
        self.loss_type = loss_type
        self.list = []
        self.key = key
        self.first_stage = CNN_VAE(**first_stage_config)
        self.train_first_stage = train_first_stage
        if not train_first_stage:
            self.first_stage.eval()
        self.norm = norm
        self.encoder = Encoder(**dd_config)
        self.decoder = Decoder(**dd_config)
        self.feature_spatial = self.first_stage.feature_spatial // math.prod(dd_config['ch_mult'])
        self.feature_dim = dd_config['z_channels']
        self.if_classify = if_classify
        if if_classify:
            self.class_block = nn.Sequential(
                AttentionPool2d(
                    spacial_dim=self.feature_spatial,
                    embed_dim=self.feature_dim,
                    num_heads=4,
                    output_dim=512
                ),
                nn.LayerNorm(512),
            )


        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, ckpt_path):
        if os.path.splitext(ckpt_path)[-1] == '.pt':
            checkpoint = torch.load(ckpt_path)
            self.load_state_dict(checkpoint['model'], strict=False)
        elif os.path.splitext(ckpt_path)[-1] == '.ckpt':
            checkpoint = torch.load(ckpt_path)
            if 'loss' in checkpoint['state_dict']:
                del checkpoint['state_dict']['loss']
            self.load_state_dict(checkpoint['state_dict'], strict=False)

        print(f"Loaded model from {ckpt_path}")


    def encode(self, x):
        """
        x: (B, C, H, W) eg: (B, 1, 1024, 1024)
        output: (B, C_out, H_out, W_out) eg: (B, 3, 64, 64)
        """
        if self.train_first_stage:
            mu, logvar = self.first_stage.encode(x)
        else:
            with torch.no_grad():
                mu, logvar = self.first_stage.encode(x)
        # x = self.first_stage.reparameterize(mu, logvar)
        x = self.encoder(mu) # (B, C, H, W) -> (B, C_out, H_out, W_out)
        mu, logvar = torch.chunk(x, 2, dim=1)
        if self.norm:
            mu = mu / (mu.norm(dim=1, keepdim=True)+1e-32)
        logvar = torch.clamp(logvar, -30, 30)
        # self.list.append(torch.sum(torch.abs(mu)).cpu().detach().numpy())
        # print('*'*10,'len:',len(self.list),'*'*10)
        # print('*'*10,'avg:',np.average(self.list),'*'*10)
        return mu, logvar

    def get_logit(self, mu):
        """
        mu: (B, C_out, H_out, W_out) eg: (B, 3, 64, 64)
        logit: (B, 512)
        """
        assert self.if_classify, "if_classify must be True"
        logit = self.class_block(mu)
        return logit

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
        x = self.decoder(z)
        if self.train_first_stage:
            x = self.first_stage.decode(x)
        else:
            with torch.no_grad():
                x = self.first_stage.decode(x)
        return x

    def sample(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return self.decode(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        z = mu
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def get_input(self, batch, k):
        if k == 'hmi_image':
            x = batch[self.key]
        elif k == 'aia0094_image':
            x = batch[self.key]
        else:
            x = batch[self.key]
            # raise NotImplementedError(f"Key {k} not supported")
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def calculate_loss(self, x, return_moment=False):
        """
        a method to get the loss only from input x
        """
        recon_x, mu, logvar = self(x)
        if self.loss_type == 'MSE':
            RECON_LOSS = F.mse_loss(recon_x, x, reduction='mean')
        elif self.loss_type == 'BCE':
            RECON_LOSS = F.binary_cross_entropy(recon_x, x, reduction='sum')
        else:
            raise ValueError(f"loss_type {self.loss_type} is not supported")
        KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))/mu.numel()

        if return_moment:
            return RECON_LOSS, KLD, mu, logvar, recon_x
        else:
            return RECON_LOSS, KLD
    

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
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt, scheduler = config_optimizers(self.learning_optimizer, self.parameters(), lr, self.learning_schedule)
        return (opt, scheduler)

    def log_images(self, batch, N=2):
        print('Begin to log images')
        log = dict()
        modals = dict()

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
        modals['inputs'] = self.vae_modal
        modals['recon'] = self.vae_modal
        modals['mu'] = self.vae_modal
        modals['samples'] = self.vae_modal

        print('Log images down')
        return log, modals
    
class CNN_VAE_three(pl.LightningModule):
    def __init__(self,vae_config,vae_train,dim,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.vae = CNN_VAE_two(**vae_config)
        if not vae_train:
            self.vae.eval()
        self.transform = transformNetwork(dim)
        self.ffn = FFN()
    
    def forward(self,x):
        x = self.vae.encode(x)
        z = self.ffn(x)
        rec_x = self.vae.decode(z)
        return z,rec_x
    
    def training_step(self,batch,batch_idx):
        x = self.vae.get_input(batch,self.vae.vae_modal)
        z,rec_x = self(x)
        loss = F.mse_loss(rec_x,x)
        self.log('train_loss',loss)
        return loss
    
    def encode(self,x):
        x , v = self.vae.encode(x)
        x = self.ffn(x)
        return x , v
    
    def decode(self,z):
        return self.vae.decode(z)
    

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class aia0094_CNN_VAE(CNN_VAE):
    def __init__(self, loss_config, **kwargs):
        super().__init__(**kwargs)

        loss_model = instantiate_from_config(loss_config)
        self.loss = loss_model.eval()
        self.loss.train = disabled_train
        for param in self.loss.parameters():
            param.requires_grad = False

    def shared_step(self, batch, batch_idx):
        x = self.get_input(batch, 'aia0094_image')
        recon_x, mu, logvar = self(x)
        posteriors = (mu, logvar)
        weights = torch.ones_like(x)

        log_split = 'train' if self.training else 'val'
        loss, loss_dict = self.loss(x, recon_x, posteriors, weights, log_split)
        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss


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