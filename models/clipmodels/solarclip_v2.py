"""This is the implementation of SolarCLIP_v2, which is a modified version of SolarCLIP. using pretrained Tokenizer (VQ-GAN or VAE)"""
from typing import Tuple, Union

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from einops import rearrange

from models.reconmodels.autoencoder.util import config_optimizers
from models.clipmodels.modules.vit import BaseVisionTransformer, Remove_class_token
from models.reconmodels.ldm.util import instantiate_from_config

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class SolarCLIP_v2(pl.LightningModule):
    def __init__(self,
                 ckpt_path: str=None,
                 base_modal_key: str='hmi_image',
                 base_modal_TokenizerConfig: dict=None,
                 base_modal_VitConfig: dict=None,
                 paired_modal_key: str='aia0094_image',
                 paired_modal_TokenizerConfig: dict=None,
                 paired_modal_VitVonfig: dict=None,
                 token_type: str='all embedding',
                 inner_loss_rate: float=0.0,
                         ):
        super().__init__()
        self.save_hyperparameters()
        self.base_modal_key = base_modal_key
        self.paired_modal_key = paired_modal_key
        self.token_type = token_type
        self.inner_loss_rate = inner_loss_rate

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.instantiate_basemodal_tokenizer(base_modal_TokenizerConfig)
        self.instantiate_pairedmodal_tokenizer(paired_modal_TokenizerConfig)
        self.instantiate_basemodal_vit(base_modal_VitConfig)
        self.instantiate_pairedmodal_vit(paired_modal_VitVonfig)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    @torch.no_grad()
    def init_from_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.load_state_dict(checkpoint, strict=False)

    def instantiate_basemodal_tokenizer(self, base_modal_config):
        if self.base_model_key == 'hmi_image':
            model = instantiate_from_config(base_modal_config)
            print(f"Base model {model.__class__.__name__} loaded")
            self.base_model = model.eval()
            self.base_model.train = disabled_train
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.tokenizer_base = self.base_model.encode
        else:
            raise NotImplementedError(f"Base model key {self.base_model_key} not supported, please choose base_model_key in ['hmi_image']")
        
    def instantiate_pairedmodal_tokenizer(self, paired_modal_config):
        model = instantiate_from_config(paired_modal_config)
        print(f"Paired model {model.__class__.__name__} loaded")
        if model.__class__.__name__ == 'ViTMAE' or model.__class__.__name__ == 'VQModel':
            self.paired_model = model.eval()
            self.paired_model.train = disabled_train
            for param in self.paired_model.parameters():
                param.requires_grad = False
            self.tokenizer_paired = self.paired_model.encode
        else:
            self.tokenizer_paired = model

    def instantiate_basemodal_vit(self, vit_config):
        self.vit_base = BaseVisionTransformer(vit_config)

    def instantiate_pairedmodal_vit(self, vit_config):
        self.vit_paired = BaseVisionTransformer(vit_config)

    def encode_base(self, x):
        with torch.no_grad():
            x = self.tokenizer_base(x)
            if len(x.shape) == 4:
                x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.vit_base(x)
        return x

    def encode_paired(self, x):
        with torch.no_grad():
            x = self.tokenizer_paired(x)
            if len(x.shape) == 4:
                x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.vit_paired(x)
        return x

    def get_input(self, batch, k):
        if k == 'hmi_image':
            x = batch[:, 0, :, :, :]
        elif k == 'aia0094_image':
            x = batch[:, 1, :, :, :]
        else:
            raise NotImplementedError(f"Key {k} not supported")
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def forward(self, x_base, x_paired, token_weight_base=None, token_weight_paired=None):
        base_features = self.encode_base(x_base)   #shape = [batch_size, length,embed_dim]
        paired_features = self.encode_paired(x_paired)
        
        # normalized features
        base_features = base_features / (base_features.norm(dim=-1, keepdim=True)+1e-32)
        paired_features = paired_features / (paired_features.norm(dim=-1, keepdim=True)+1e-32)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()

        if self.token_type == 'class embedding':
            logits_per_base = logit_scale * base_features @ paired_features.t()
            logits_per_paired = logits_per_base.t()
            inner_cor_matrix = None

        elif self.token_type == 'all embedding':
            B, L = base_features.shape[0], base_features.shape[1]
            if token_weight_base is None:
                token_weight_base = torch.ones([B,L],dtype = base_features.dtype,device = base_features.device)
            if token_weight_paired is None:
                token_weight_paired = torch.ones([B,L],dtype = paired_features.dtype,device = paired_features.device)
            assert (token_weight_base.shape == (B, L) and token_weight_paired.shape == (B, L)) # [B,L] tensor
            token_weight_base = token_weight_base.unsqueeze(-1)
            token_weight_paired = token_weight_paired.unsqueeze(-1)

            base_features = torch.einsum('BLD,BLd->BLD', base_features, token_weight_base)
            paired_features = torch.einsum('BLD,BLd->BLD', paired_features, token_weight_paired)
            inner_cor_matrix = torch.einsum('BLD,BlD->BLl', base_features, paired_features)
            inner_cor_matrix = inner_cor_matrix.mean(dim=0) # [L,L]
            cor_matrix = torch.einsum('BLD,bLD->BbL', base_features, paired_features)
            cor_matrix = cor_matrix.mean(dim=-1) # [B,B]

            logits_per_base = logit_scale * cor_matrix
            logits_per_paired = logits_per_base.t()
            inner_cor_matrix = logit_scale * inner_cor_matrix

        return logits_per_base, logits_per_paired, inner_cor_matrix

    def loss_function(self, logits_per_base, logits_per_paired, inner_cor_matrix, inner_loss_rate = 0, token_weight_base = None, token_weight_paired = None, criterion = nn.functional.cross_entropy):

        ground_truth = torch.arange(logits_per_base.shape[0], dtype=torch.long, device=logits_per_base.device)
        
        loss_base = criterion(logits_per_base, ground_truth)
        loss_paired = criterion(logits_per_paired, ground_truth)
        loss = (loss_base + loss_paired) / 2
        acc = (torch.argmax(logits_per_base, dim=1) == ground_truth).float().mean().item()

        assert inner_loss_rate >=0
        if inner_loss_rate > 0:
            ground_truth = torch.arange(inner_cor_matrix.shape[-1], dtype=torch.long, device=inner_cor_matrix.device)#[l]
            # ground_truth = ground_truth.unsqueeze(0).expand(inner_cor_matrix.shape[0],-1) # [L,L]
            loss_inner = criterion(inner_cor_matrix,ground_truth)/2
            loss_inner = loss_inner + criterion(inner_cor_matrix.t(),ground_truth)/2
        else:
            loss_inner = torch.tensor(0, dtype=torch.float32, device=inner_cor_matrix.device)

        return loss, loss_inner, acc
    
    def get_loss(self, logits_per_base, logits_per_paired, inner_cor_matrix, inner_loss_rate = 0., token_weight_base = None, token_weight_paired = None, criterion = torch.nn.functional.cross_entropy):
        log_prefix = 'train' if self.training else 'val'
        loss_dict = {}

        loss, loss_inner, acc = self.loss_function(logits_per_base, logits_per_paired, inner_cor_matrix, inner_loss_rate = inner_loss_rate, token_weight_base = token_weight_base, token_weight_paired = token_weight_paired, criterion = criterion)
        loss_dict.update({f'{log_prefix}/loss': loss})
        loss_dict.update({f'{log_prefix}/loss_inner': loss_inner})
        loss_dict.update({f'{log_prefix}/acc': acc})
        
        return loss, loss_dict
    
    def shared_step(self, batch, batch_idx):
        base_modal_data = self.get_input(batch, self.base_modal_key)
        paired_modal_data = self.get_input(batch, self.paired_modal_key)

        logits_per_base, logits_per_paired, inner_cor_matrix = self(base_modal_data, paired_modal_data, token_weight_base=None, token_weight_paired=None)

        loss, loss_dict = self.get_loss(logits_per_base, logits_per_paired, inner_cor_matrix, inner_loss_rate = self.inner_loss_rate, token_weight_base = None, token_weight_paired = None, criterion = torch.nn.functional.cross_entropy)
        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt, scheduler = config_optimizers(self.learning_optimizer, self.parameters(), lr, self.learning_schedule)
        return (opt, scheduler)
    
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
            inputs = self.get_input(batch, self.input_modal_key)
            targets = inputs.clone()
            inputs = inputs.to(self.device)
        N = min(N, inputs.shape[0])
        log['inputs'] = inputs[:N]
        log['targets'] = targets[:N]
        self.eval()
        with torch.no_grad():
            targets_token_hat_mask_ratio_set = self(inputs, mask_ratio=self.mask_ratio)[0]
            targets_hat_mask_ratio_set = self.unpatchify(targets_token_hat_mask_ratio_set)
            targets_token_hat_mask_ratio_0 = self(inputs, mask_ratio=0.)[0]
            targets_hat_mask_ratio_0 = self.unpatchify(targets_token_hat_mask_ratio_0)
            targets_token_hat_inference = self.forward_inference(inputs)
            targets_hat_inference = self.unpatchify(targets_token_hat_inference)
        log['targets_hat_mask_ratio_set'] = targets_hat_mask_ratio_set[:N]
        log['targets_hat_mask_ratio_0'] = targets_hat_mask_ratio_0[:N]
        log['targets_hat_inference'] = targets_hat_inference[:N]
        self.train()

        modals['inputs'] = self.input_modal_key
        modals['targets'] = self.input_modal_key
        modals['targets_hat_mask_ratio_set'] = self.input_modal_key
        modals['targets_hat_mask_ratio_0'] = self.input_modal_key
        modals['targets_hat_inference'] = self.input_modal_key
        print('image log done')
        return log, modals
    

class SolarCLIP_remove_CLS(nn.Module):
    def __init__(self,
                 modal_key,
                 solarclip_config):
        super().__init__()
        self.modal_key = modal_key
        self.instantiate_solarclip(solarclip_config)

    def instantiate_solarclip(self, config):
        model = instantiate_from_config(config)
        self.solarclip = model.eval()
        self.solarclip.train = disabled_train
        for param in self.solarclip.parameters():
            param.requires_grad = False

    def encode_clip(self, x):
        if self.modal_key == 'hmi':
            return self.solarclip.encode_hmi(x)
        elif self.modal_key == 'aia0094_image':
            return self.solarclip.encode_aia(x)
        else:
            raise ValueError('Invalid modal key')

    def forward(self, x):
        x = self.encode_clip(x)
        x = Remove_class_token()(x)
        return x