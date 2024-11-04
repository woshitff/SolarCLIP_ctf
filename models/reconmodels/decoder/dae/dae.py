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

class ReshapeTo2D(nn.Module):
    def __init__(self, h=16, w=16):
        super().__init__()
        self.h = h
        self.w = w

    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h=self.h, w=self.w)
    
class LinearProjectionToImage(nn.Module):
    def __init__(self, input_dim=(256, 16, 16), output_dim=(1, 256, 256)):
        super(LinearProjectionToImage, self).__init__()
        input_size = input_dim[0] * input_dim[1] * input_dim[2]  # 768 * 16 * 16
        output_size = output_dim[0] * output_dim[1] * output_dim[2]  # 3 * 64 * 64
        self.output_dim = output_dim
        self.fc = nn.Linear(input_size, output_size)
        self.activation = nn.laynorm(output_size)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  
        x = self.fc(x)             
        x = self.activation(x)      
        return x.reshape(x.size(0), self.output_dim[0], self.output_dim[1], self.output_dim[2])
    

class SolarCLIPDAE(pl.LightningModule):
    """Get image embedding from SolarCLIP and project it to image space."""
    def __init__(self, 
                 solarclip_config,
                 decode_modal_key='aia0094_image', 
                 projection_type='Linear', 
                 loss_type='l2',
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.decode_modal_key = decode_modal_key
        self.projection_type = projection_type
        self.loss_type = loss_type

        self._init_solarclip(solarclip_config)
        self.linear_projection = nn.Linear(768, 256)
        self.ReshapeProjection = self._get_projection(projection_type)

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
        
    def _get_projection(self, projection_type):
        projectin_options = {
            "ConvTrans": nn.Sequential(
                ReshapeTo2D(16, 16),
                nn.ConvTranspose2d(768, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(16, 128),
                nn.Tanh(),
                nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(1, 3),
                nn.Tanh()
                ),
            "Linear": nn.Sequential(
                self.linear_projection,
                ReshapeTo2D(16, 16),
                LinearProjectionToImage(input_dim=(256, 16, 16), output_dim=(1, 256, 256))
                )
        }

        if projection_type in projectin_options:
            return projectin_options[projection_type]
        else:
            raise ValueError(f"Unknown projection type {projection_type}")

    def forward(self, x):
        x = self.solarclip(x)
        x = Remove_class_token()(x) # (B, 257, 768) -> (B, 256, 768)
        x = self.ReshapeProjection(x) # (B, 256, 768) -> (B, 1, 256, 256)
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
        x = transforms.Resize(size=256)(x)
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
            inputs = self.get_input(batch[:, 1, :, :, :], self.decode_modal_key)
            targets = self.get_target(batch[:, 1, :, :, :], self.decode_modal_key)

        N = min(N, inputs.shape[0])
        log['inputs'] = inputs[:N]
        log['targets'] = targets[:N]
        self.eval()
        with torch.no_grad():
            targets_hat = self(inputs)
        log['targets_hat'] = targets_hat[:N]
        self.train()

        modals['inputs'] = self.inputs_modal
        modals['targets'] = self.targets_modal
        modals['targets_hat'] = self.targets_modal

        return log, modals