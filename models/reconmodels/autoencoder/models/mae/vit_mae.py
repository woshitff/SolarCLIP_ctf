# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
import pytorch_lightning as pl

from models.reconmodels.autoencoder.util import config_optimizers
from models.reconmodels.autoencoder.util import instantiate_from_config
from models.reconmodels.autoencoder.models.mae.util.pos_embed import get_2d_sincos_pos_embed


class ViTMAE(pl.LightningModule):
    """Vision Transformer with Masked Autoencoder (ViTMAE) get encode of shape (B, L+1, D)"""
    def __init__(self,
                 ckpt_path=None,
                 input_modal_key='hmi_image',
                 mask_ratio=0.5,
                 img_size=1024, patch_size=64, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,):
        super().__init__()
        self.save_hyperparameters()
        self.input_modal_key = input_modal_key
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.norm_pix_loss = norm_pix_loss

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        self.initialize_weights()

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    @torch.no_grad()
    def init_from_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.load_state_dict(checkpoint, strict=False)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 *C)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        c = imgs.shape[1]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *c)
        imgs: (N, c, H, W)
        """
        p = self.patch_size
        c = self.in_chans
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        # print(f'x shape after decoder embed: {x.shape}')

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)

        x = x[:, 1:, :]

        return x

    def forward(self, imgs, mask_ratio=0.):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        # print(f'latent shape: {latent.shape}')
        pred = self.forward_decoder(latent, ids_restore)  
        return pred, mask
    
    def encode(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
    
    def decode(self, x):
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x
    
    def forward_inference(self, x):
        x = self.encode(x)
        x = self.decode(x)
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
    
    def loss_function(self, pred, imgs, mask):
        """
        imgs: [N, C, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def get_loss(self, pred, x, mask):
        log_prefix = 'train' if self.training else 'val'
        loss_dict = {}

        loss = self.loss_function(pred, x, mask)
        loss_dict.update({f'{log_prefix}/loss': loss})
        
        return loss, loss_dict
    
    def shared_step(self, batch, batch_idx):
        x = self.get_input(batch, self.input_modal_key)
        mask_ratio = self.mask_ratio
        pred, mask = self(x, mask_ratio)
        loss, loss_dict = self.get_loss(pred, x, mask)
        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
    
if __name__ == '__main__':
    pass