import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pytorch_lightning as pl
from torchvision import transforms

from models.clipmodels.modules.vit import Remove_class_token
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
    def __init__(self, in_channels, out_channels, bilinear=False):
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
    

class ClipCNNDecoder(pl.LightningModule):
    """Get image embedding from SolarCLIP and project it to image space."""
    def __init__(self, 
                 solarclip_config,
                 decode_modal_key='aia0094_image', 
                 layer_list=[2, 2, 2],
                 in_channels=768,
                 hidden_channels=512,
                 out_channels=1,
                 loss_type='l2',
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.solarclip_config = solarclip_config
        self.decode_modal_key = decode_modal_key
        self.layer_list = layer_list
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.loss_type = loss_type
        self.out_size = self.solarclip_config.params.image_resolution_hmi // 2**len(layer_list)

        in_out = []
        for i in range(len(layer_list)):
            in_c = hidden_channels // 2**i
            out_c = hidden_channels // 2**(i+1)
            in_out.append((in_c, out_c))
        self.in_out = in_out
        assert all(in_c % 2 == 0 and out_c % 2 == 0 for in_c, out_c in in_out), "All channels must be multiples of 2" 

        self._init_solarclip(solarclip_config)

        self.blocks = nn.ModuleList()
        for i, num_blocks in enumerate(layer_list):
            self.blocks.append(nn.Sequential(
                *nn.ModuleList(
                [ResidualConvBlock(in_out[i][0], in_out[i][0]) for _ in range(num_blocks)]
                ),
                Upsample(in_out[i][0], in_out[i][1]),
            ))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.GroupNorm(hidden_channels//16, hidden_channels),
            *self.blocks,
            nn.GroupNorm(hidden_channels//(2**len(layer_list)*16), hidden_channels//2**(len(layer_list))),
            nn.ELU(),
            nn.Conv2d(in_out[len(layer_list)-1][1], 1, kernel_size=3, padding=1)
        )

    def _init_solarclip(self, solarclip_config, freeze=True):
        solarclip = instantiate_from_config(solarclip_config)
        if freeze:
            self.solarclip = solarclip.eval()
            self.solarclip.train = disabled_train
            for param in self.solarclip.parameters():
                param.requires_grad = False
        if self.decode_modal_key == 'magnet_image':
            self.solarclip = self.solarclip.visual_hmi
        elif self.decode_modal_key == 'aia0094_image':
            self.solarclip = self.solarclip.visual_aia
        else:
            raise ValueError(f"Unknown embedding key {self.decode_modal_key}")
        
    def get_cliptoken(self, x):
        x = self.solarclip(x)
        x = Remove_class_token()(x)
        return x
    
    def decode(self, x):
        # (B, 256, 768) -> (B, 256, 16, 16) -> (B, 1, 128, 128)
        x = rearrange(x, 'b (h w) c -> b c h w', h=16, w=16)
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = self.get_cliptoken(x)
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
        x = transforms.Resize(self.out_size)(x)
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