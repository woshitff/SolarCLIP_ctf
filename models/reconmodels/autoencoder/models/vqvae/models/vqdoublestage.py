from packaging import version

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from contextlib import contextmanager
from einops import rearrange

from models.reconmodels.autoencoder.models.vqvae.modules.taming_vqgan.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from models.reconmodels.autoencoder.models.vqvae.modules.model import Encoder, Decoder
from models.reconmodels.autoencoder.util import instantiate_from_config
from models.reconmodels.autoencoder.util import config_optimizers
from models.reconmodels.ldm.modules.ema import LitEma


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class VQDoubleStageModel(pl.LightningModule):
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
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        # self.automatic_optimization = False
        self.first_vq_model_config = first_vq_model_config
        self.ddconfig = ddconfig
        self.lossconfig = lossconfig
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
        first_vq_dim = self.first_vq_model.embed_dim
        first_vq_codebook_size = self.first_vq_model.n_embed
        self.mlp =   torch.nn.Sequential(
            torch.nn.Linear(first_vq_dim, 4*first_vq_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4*first_vq_dim, 4*first_vq_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4*first_vq_dim, first_vq_codebook_size),
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
        
        print('model init down')
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
            # if self.first_vq_model.hparams.get("return_indices", True):
            quant_first, ind_first = self.first_vq_model.encode(x)
            return quant_first, ind_first
            # else:
            #     quant_first = self.first_vq_model.encode(x)
            #     return quant_first

    def decode_first_vqmodel(self, h):
        with torch.no_grad():
            xrec = self.first_vq_model.decode(h)
        return xrec

    def convert_logits_to_features(self, logits):
        """
        将 logits 转换为特征向量，首先从 logits 中获取类别索引，然后
        查询 codebook 中的特征向量，并调整为指定的形状 (b, c, h, w)。
        
        Args:
            logits (Tensor): 形状为 (b, d)，表示类别预测的 logits。
        
        Returns:
            Tensor: 重构后的特征向量，形状为 (b, c, h, w)。
        """
        to_first_vq_indices = logits.argmax(dim=-1)
        # to_first_vq_indices = rearrange(to_first_vq_indices, "(b h w) c -> b c h w")
        
        first_vqcodebook_features = self.first_vq_model.quantize.embedding(to_first_vq_indices)  
        h = self.first_vq_model.ddconfig["resolution"] // 2**(len(self.first_vq_model.ddconfig["ch_mult"]) - 1)
        reconstructed_to_first_vq = rearrange(first_vqcodebook_features, "(b h w) c -> b c h w", h=h, w=h)

        return reconstructed_to_first_vq

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
            dec_flatten = rearrange(dec, "b c h w -> (b h w) c")
            logits = self.mlp(dec_flatten)
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
            # print(f"Using first VQ model to encode {k}.")
            quant_first, ind_first = self.encode_first_vqmodel(origin_inputs)
        # print('get input down')
        return origin_inputs, quant_first, ind_first

    def training_step(self, batch, batch_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        _, quant_first, ind_first = self.get_input(batch, self.vq_modal) # ind_first shape (b*h*w,)
        xrec, qloss, ind_second, logits = self(quant_first, return_pred_indices=True)
        loss, log_dict = self.loss(qloss, quant_first, xrec, 
                                ind_first, logits,
                                split="train",predicted_indices=ind_second)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        _, quant_first, ind_first = self.get_input(batch, self.vq_modal)
        xrec, qloss, ind_second, logits = self(quant_first, return_pred_indices=True)
        loss, log_dict = self.loss(qloss, quant_first, xrec, 
                                ind_first, logits,
                                split="val"+suffix, predicted_indices=ind_second)

        rec_loss = log_dict[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log(f"val{suffix}/aeloss", loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        # from metrics.reconmodels.autoencoder.vqvae.vqgan import FID
        # fid = FID().calculate_fid(quant_first, xrec)
        # self.log(f"val{suffix}/fid", fid,
        #            prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt, scheduler = config_optimizers(self.learning_optimizer, self.parameters(), lr, self.learning_schedule)
        return (opt, scheduler)

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        modals = dict()

        original_inputs, quantized_first_vq, quantization_indices_first = self.get_input(batch, self.vq_modal)
        original_inputs = original_inputs.to(self.device)
        quantized_first_vq = quantized_first_vq.to(self.device)

        if only_inputs:
            log["original_inputs"] = original_inputs
            log["quantized_first_vq"] = quantized_first_vq
            modals['origin_inputs'] = self.vq_modal
            modals['quantized_first_vq'] = self.vq_modal
            return log, modals
        
        self.eval()
        with torch.no_grad():
            # Perform the encoding and quantization steps
            prequant_second_vq = self.encode_to_prequant(quantized_first_vq)
            quantized_second_vq, _, _ = self.quantize(prequant_second_vq)

            # Perform reconstruction
            reconstructed_second_vq, _, _, logits = self(quantized_first_vq, return_pred_indices=True)
            reconstructed_to_first_vq = self.convert_logits_to_features(logits)
            reconstructed_to_original = self.decode_first_vqmodel(reconstructed_to_first_vq)

        self.train()

        log["original_inputs"] = original_inputs
        log["quantized_first_vq"] = quantized_first_vq
        log['prequant_second_vq'] = prequant_second_vq
        log["quantized_second_vq"] = quantized_second_vq
        log["reconstructed_second_vq"] = reconstructed_second_vq
        log["reconstructed_to_first_vq"] = reconstructed_to_first_vq
        log["reconstructed_to_original"] = reconstructed_to_original

        # if plot_ema:
        #     with self.ema_scope():
        #         xrec_ema, _ = self(x)
        #         if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
        #         log["reconstructions_ema"] = xrec_ema

        modals['original_inputs'] = self.vq_modal
        modals['quantized_first_vq'] = self.vq_modal
        modals['prequant_second_vq'] = self.vq_modal
        modals['quantized_second_vq'] = self.vq_modal
        modals['reconstructed_second_vq'] = self.vq_modal
        modals['reconstructed_to_first_vq'] = self.vq_modal
        modals['reconstructed_to_original'] = self.vq_modal
        
        print('Load image down')
        return log, modals
    

class VQTokenizer(VQDoubleStageModel):
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
