from packaging import version
from math import log2
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl

from models.reconmodels.autoencoder.util import instantiate_from_config

# see https://github.com/vvvm23/vqvae-2/blob/main/vqvae.py

class ReZero(nn.Module):
    def __init__(self, in_channels: int, res_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x) * self.alpha + x

class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, res_channels: int, nb_layers: int):
        super().__init__()
        self.stack = nn.Sequential(*[ReZero(in_channels, res_channels) 
                        for _ in range(nb_layers)
                    ])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.stack(x)

class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, res_channels: int, nb_res_layers: int, downscale_factor: int):
        super().__init__()
        assert log2(downscale_factor) % 1 == 0, "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))
        layers = []
        c_channel, n_channel = in_channels, hidden_channels // 2
        for _ in range(downscale_steps):
            layers.append(nn.Sequential(
                nn.Conv2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, hidden_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        layers.append(ResidualStack(n_channel, res_channels, nb_res_layers))

        self.layers = nn.Sequential(*layers)

        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, 
            in_channels: int, hidden_channels: int, out_channels: int,
            res_channels: int, nb_res_layers: int,
            upscale_factor: int,
        ):
        super().__init__()
        assert log2(upscale_factor) % 1 == 0, "Downscale must be a power of 2"
        upscale_steps = int(log2(upscale_factor))
        layers = [nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1)]
        layers.append(ResidualStack(hidden_channels, res_channels, nb_res_layers))
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        for _ in range(upscale_steps):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, out_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        # layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

"""
    Almost directly taken from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
    No reason to reinvent this rather complex mechanism.

    Essentially handles the "discrete" part of the network, and training through EMA rather than 
    third term in loss function.
"""
class CodeLayer(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, nb_entries: int):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, embed_dim, 1)

        self.dim = embed_dim
        self.n_embed = nb_entries
        self.decay = 0.99
        self.eps = 1e-5

        embed = torch.randn(embed_dim, nb_entries, dtype=torch.float32)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(nb_entries, dtype=torch.float32))
        self.register_buffer("embed_avg", embed.clone())

    # @torch.amp.autocast(enabled=False)
    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, float, torch.LongTensor]:
        x = self.conv_in(x.float()).permute(0,2,3,1)
        flatten = x.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # TODO: Replace this? Or can we simply comment out?
            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        return quantize.permute(0, 3, 1, 2), diff, embed_ind

    def embed_code(self, embed_id: torch.LongTensor) -> torch.FloatTensor:
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class Upscaler(nn.Module):
    def __init__(self,
            embed_dim: int,
            scaling_rates: list[int],
        ):
        super().__init__()
        self.stages = nn.ModuleList()
        for sr in scaling_rates:
            upscale_steps = int(log2(sr))
            layers = []
            for _ in range(upscale_steps):
                layers.append(nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(embed_dim))
                layers.append(nn.ReLU(inplace=True))
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x: torch.FloatTensor, stage: int) -> torch.FloatTensor:
        return self.stages[stage](x)

"""
    Main VQ-VAE-2 Module, capable of support arbitrary number of levels
    TODO: A lot of this class could do with a refactor. It works, but at what cost?
    TODO: Add disrete code decoding function
"""
class VQVAE2(pl.LightningModule):
    def __init__(self,
            vq_modal: str                  = 'hmi_image',
            in_channels: int                = 1,
            hidden_channels: int            = 128,
            res_channels: int               = 32,
            nb_res_layers: int              = 2,
            nb_levels: int                  = 2,
            embed_dim: int                  = 8,
            nb_entries: int                 = 8192,
            scaling_rates: list[int]        = [8, 2],
            lossconfig: dict                = None,
            ckpt_path: str                  = None,
            ignore_keys: list[str]          = [],
            scheduler_config: dict          = None,
            lr_g_factor=1.0,
        ):
        super().__init__()
        self.automatic_optimization = False
        self.vq_modal = vq_modal

        self.nb_levels = nb_levels
        assert len(scaling_rates) == nb_levels, "Number of scaling rates not equal to number of levels!"

        self.encoders = nn.ModuleList([Encoder(in_channels, hidden_channels, res_channels, nb_res_layers, scaling_rates[0])])
        for i, sr in enumerate(scaling_rates[1:]):
            self.encoders.append(Encoder(hidden_channels, hidden_channels, res_channels, nb_res_layers, sr))

        self.codebooks = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.codebooks.append(CodeLayer(hidden_channels+embed_dim, embed_dim, nb_entries))
        self.codebooks.append(CodeLayer(hidden_channels, embed_dim, nb_entries))

        self.decoders = nn.ModuleList([Decoder(embed_dim*nb_levels, hidden_channels, in_channels, res_channels, nb_res_layers, scaling_rates[0])])
        for i, sr in enumerate(scaling_rates[1:]):
            self.decoders.append(Decoder(embed_dim*(nb_levels-1-i), hidden_channels, embed_dim, res_channels, nb_res_layers, sr))

        self.upscalers = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.upscalers.append(Upscaler(embed_dim, scaling_rates[1:len(scaling_rates) - i][::-1]))

        self.loss = instantiate_from_config(lossconfig)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

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

    def forward(self, x):
        encoder_outputs = []
        code_outputs = []
        decoder_outputs = []
        upscale_counts = []
        id_outputs = []
        diffs = []

        for enc in self.encoders:
            if len(encoder_outputs):
                encoder_outputs.append(enc(encoder_outputs[-1]))
            else:
                encoder_outputs.append(enc(x))

        for l in range(self.nb_levels-1, -1, -1):
            codebook, decoder = self.codebooks[l], self.decoders[l]

            if len(decoder_outputs): # if we have previous levels to condition on
                code_q, code_d, emb_id = codebook(torch.cat([encoder_outputs[l], decoder_outputs[-1]], axis=1))
            else:
                code_q, code_d, emb_id = codebook(encoder_outputs[l])
            diffs.append(code_d)
            id_outputs.append(emb_id)

            code_outputs = [self.upscalers[i](c, upscale_counts[i]) for i, c in enumerate(code_outputs)]
            upscale_counts = [u+1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1)))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        return decoder_outputs[-1], diffs, encoder_outputs, decoder_outputs, id_outputs
    
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

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.vq_modal)
        xrec, qloss,  _, _, _ = self(x)
        qloss = sum(qloss)

        opt_g, opt_d = self.optimizers()

        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train",
                                        )
        opt_g.zero_grad()
        self.manual_backward(aeloss)
        opt_g.step()
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        # discriminator
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        opt_d.zero_grad()
        self.manual_backward(discloss)
        opt_d.step()
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        # with self.ema_scope():
        #     log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.vq_modal)
        xrec, qloss, _, _, _ = self(x)
        qloss = sum(qloss)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        from metrics.reconmodels.autoencoder.vqvae.vqgan import FID
        print(f"x.shape: {x.shape}, xrec.shape: {xrec.shape} ")
        try:
            fid = FID().calculate_fid(x, xrec)
            self.log(f"val{suffix}/fid", fid,
                    prog_bar=True, logger=True, on_step=True, on_epoch=True)
            print(f'FID Value: {fid}')
        except:
            print("FID calculation failed.")

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
        opt_ae = torch.optim.Adam(list(self.encoders.parameters())+
                                  list(self.decoders.parameters())+
                                  list(self.codebooks.parameters())+
                                  list(self.upscalers.parameters()),
                                #   list(self.post_quant_conv.parameters()),
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
        return self.decoders[-1].layers[-1].weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        print('begin log image')
        log = dict()
        modals = dict()

        x = self.get_input(batch, self.vq_modal)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        
        self.eval()
        with torch.no_grad():
            xrec, _, _, _, _ = self(x)
        self.train()

        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            
        log["inputs"] = x
        log["recon"] = xrec

        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema

        modals["inputs"] = self.vq_modal
        modals["recon"] = self.vq_modal
        print('log image done')
        return log, modals


    def decode_codes(self, *cs):
        decoder_outputs = []
        code_outputs = []
        upscale_counts = []

        for l in range(self.nb_levels - 1, -1, -1):
            codebook, decoder = self.codebooks[l], self.decoders[l]
            code_q = codebook.embed_code(cs[l]).permute(0, 3, 1, 2)
            code_outputs = [self.upscalers[i](c, upscale_counts[i]) for i, c in enumerate(code_outputs)]
            upscale_counts = [u+1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1)))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        return decoder_outputs[-1]

