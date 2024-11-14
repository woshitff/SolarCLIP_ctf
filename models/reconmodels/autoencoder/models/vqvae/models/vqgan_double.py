from packaging import version

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from contextlib import contextmanager

from models.reconmodels.autoencoder.models.vqvae.modules.taming_vqgan.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from models.reconmodels.autoencoder.util import instantiate_from_config
from models.reconmodels.autoencoder.models.vae.CNN_VAE_v2 import Encoder, Decoder
from models.reconmodels.ldm.modules.ema import LitEma


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class VQVAE2Model(pl.LightningModule):
    def __init__(self,
                 first_vq_model_config,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 vq_modal="hmi_image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.automatic_optimization = False
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.vq_modal = vq_modal

        self.instantiate_first_vqmodel(first_vq_model_config)
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.mlp =   torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 4*embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4*embed_dim, 4*embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4*embed_dim, self.n_embed),
            torch.nn.Softmax(dim=-1),
        )

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def instantiate_first_vqmodel(self, first_vq_model_config):
        model = instantiate_from_config(first_vq_model_config)
        self.first_vq_model = model.eval()
        self.first_vq_model.train = disabled_train
        for param in self.first_vq_model.parameters():
            param.requires_grad = False

    def encode_first_vqmodel(self, x):
        with torch.no_grad():
            if self.first_vq_model.hparams.get("return_indices", True):
                quant_first, ind_first = self.first_vq_model.encode(x)
                return quant_first, ind_first
            else:
                quant_first = self.first_vq_model.encode(x)
                return quant_first

    def decode_first_vqmodel(self, h):
        with torch.no_grad():
            xrec = self.first_vq_model.decode(h)
        return xrec

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info
    
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def forward(self, input, return_pred_indices=False):
        quant_second, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant_second)
        if return_pred_indices:
            logits = self.mlp(dec)
            return dec, diff, ind, logits
        return dec, diff

    def get_input(self, batch, k):
        if k == 'hmi_image':
            origin_inputs = batch[:, 0, :, :, :]
        elif k == 'aia0094_image':
            origin_inputs = batch[:, 1, :, :, :]
        else:
            raise NotImplementedError(f"Key {k} not supported")
        if len(origin_inputs.shape) == 3:
            origin_inputs = origin_inputs[..., None]
        origin_inputs = origin_inputs.to(memory_format=torch.contiguous_format).float()
        
        if self.first_vq_model is not None:
            print(f"Using first VQ model to encode {k}.")
            quant_first, ind_first = self.encode_first_vqmodel(origin_inputs)

        return origin_inputs, quant_first, ind_first

    def training_step(self, batch, batch_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        _, quant_first, ind_first = self.get_input(batch, self.vq_modal)
        xrec, qloss, ind_second, logits = self(quant_first, return_pred_indices=True)

        unique_indices, _ = torch.unique(ind_second, return_counts=True)
        codebook_usage = 100 * unique_indices.numel()/(batch.size(0)*self.n_embed)
        self.log("train/codebook_usage", codebook_usage, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        opt_g, opt_d = self.optimizers()

        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, quant_first, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train",
                                        predicted_indices=ind_second)
        opt_g.zero_grad()
        self.manual_backward(aeloss)
        opt_g.step()
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        # discriminator
        discloss, log_dict_disc = self.loss(qloss, quant_first, xrec, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        opt_d.zero_grad()
        self.manual_backward(discloss)
        opt_d.step()
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.vq_modal)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        from metrics.reconmodels.autoencoder.vqvae.vqgan import FID
        fid = FID().calculate_fid(x, xrec)
        self.log(f"val{suffix}/fid", fid,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        modals = dict()

        x = self.get_input(batch, self.vq_modal)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        
        self.eval()
        with torch.no_grad():
            latent_prequant = self.encode_to_prequant(x)
            latent_quant, _, _ = self.quantize(latent_prequant)
            xrec = self.decode(latent_quant)
        self.train()

        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            
        log["inputs"] = x
        log["recon"] = xrec
        log["latent_prequant"] = latent_prequant
        log["latent_quant"] = latent_quant

        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema

        modals["inputs"] = self.vq_modal
        modals["recon"] = self.vq_modal
        modals["latent_prequant"] = self.vq_modal
        modals["latent_quant"] = self.vq_modal

        return log, modals

    def to_rgb(self, x):
        assert self.vq_modal == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
    

class VQTokenizer(VQVAE2Model):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    # @torch.no_grad()
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        return quant

    # @torch.no_grad()
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
