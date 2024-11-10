import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pytorch_lightning as pl
from torchvision import transforms

from models.clipmodels.solarclip import SolarCLIP_remove_CLS
from models.clipmodels.modules.vit import Remove_class_token, Transformer
from models.reconmodels.ldm.util import instantiate_from_config

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
    
class ResidualConvBlock(nn.Module):
    # see https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, mid_channels),
            nn.ELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.ELU()
        )

    def forward(self, x):
        res = self.res_conv(x)
        x = self.double_conv(x)
        return x + res    
    
class Upsample(nn.Module):
    def __init__(self, in_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        return self.up(x)
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, bilinear=False, num_groups=16):
        super().__init__()
        self.in_channels = in_channels
        assert in_channels % 2 == 0, 'in_channels must be even'
        self.out_channels = in_channels // 2
        self.num_groups = num_groups

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.up = Upsample(in_channels, bilinear=bilinear)
        self.block = nn.Sequential(
            self.up,
            nn.ELU(),
            nn.GroupNorm(num_groups, self.out_channels)
        )

    def forward(self, x):
        return self.block(x)


class ClipVitDecoder(pl.LightningModule):
    """Get image embedding from SolarCLIP and project it to image space."""
    def __init__(self, 
                 ckpt_path=None,
                 decode_modal_key='aia0094_image', 
                 clip_config = None,
                 width=768,
                 layers=12,
                 heads=12,
                 num_upblocks = 3,
                 out_channels = 1,
                 loss_type = 'l2'
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.decode_modal_key = decode_modal_key
        self.clip_config = clip_config
        self.loss_type = loss_type

        self.instantiate_solarclip_remove_CLS(clip_config)
        scale = width ** -0.5
        self.scale = scale
        self.positional_embedding = nn.Parameter(scale * torch.randn((16) ** 2, width))
        self.transformer = Transformer(width, layers, heads)
        self.decoder = nn.Sequential(*[UpsampleBlock(768 // 2**i) for i in range(num_upblocks)])
        in_channels = 768 // 2**(num_upblocks)
        self.out = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    @torch.no_grad()
    def init_from_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        # if 'loss' in checkpoint['state_dict']:
        #     del checkpoint['state_dict']['loss']
        # self.load_state_dict(checkpoint['state_dict'], strict=False)
        self.load_state_dict(checkpoint, strict=False)

    def instantiate_solarclip_remove_CLS(self, config):
        model = instantiate_from_config(config)
        self.solarclip_remove_cls = model.eval()
        self.solarclip_remove_cls.train = disabled_train
        for param in self.solarclip_remove_cls.parameters():
            param.requires_grad = False

    def encode(self, x):
        # (B, 1, 1024, 1024) -> (B, 257, 768) -> (B, 256, 768) -> (B, 768, 256)
        x = self.solarclip_remove_cls(x)
        x = rearrange(x, 'b l d -> b d l')
        return x
    
    def decode(self, x):
        # (B, 768, 256) -> (B, 256, 768) -> (B, 256, 16, 16) -> (B, 1, 128, 128)
        x = rearrange(x, 'b d l -> b l d')
        # x = x + self.positional_embedding.to(x.dtype)
        x = self.transformer(x) # (B, 256, 768) -> (B, 256, 768)
        x = rearrange(x, 'b (h w) c -> b c h w', h=16, w=16) # (B, 256, 768) -> (B, 768, 16, 16)
        x = self.decoder(x)
        x = self.out(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def get_input(self, batch, k):
        if k == 'magnet_image':
            x = batch[:, 0, :, :, :]
        elif k == 'aia0094_image':
            x = batch[:, 1, :, :, :]
        else:
            raise NotImplementedError(f"Key {k} not supported")
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def get_target(self, batch, k):
        if k == 'magnet_image':
            x = batch[:, 0, :, :, :]
        elif k == 'aia0094_image':
            x = batch[:, 1, :, :, :]
        else:
            raise NotImplementedError(f"Key {k} not supported")
        if len(x.shape) == 3:
            x = x[..., None]
        x = transforms.Resize(size=128)(x)
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

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
        x = self.get_input(batch, self.decode_modal_key) 
        y = self.get_target(batch, self.decode_modal_key)
        y_hat = self(x)
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

    def log_images(self, batch, N=2):
        """
        Log a batch of images to tensorboard and local

        output:
            log: dictionary of images to log
            modals: dictionary of modalities determing the cmap and vmin/vmax for each image
        """
        log = dict()
        modals = dict()

        with torch.no_grad():
            inputs = self.get_input(batch, self.decode_modal_key)
            targets = self.get_target(batch, self.decode_modal_key)
        N = min(N, inputs.shape[0])
        log['inputs'] = inputs[:N]
        log['targets'] = targets[:N]
        self.eval()
        with torch.no_grad():
            targets_hat = self(inputs)
        log['targets_hat'] = targets_hat[:N]
        self.train()

        modals['inputs'] = self.decode_modal_key
        modals['targets'] = self.decode_modal_key
        modals['targets_hat'] = self.decode_modal_key
        print('image log done')
        return log, modals
    
