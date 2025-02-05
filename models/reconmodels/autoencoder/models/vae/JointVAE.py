# import math
# import os

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pytorch_lightning as pl

# from models.reconmodels.autoencoder.util import config_optimizers
# from models.reconmodels.autoencoder.util import instantiate_from_config

# class JointVAE(pl.LightningModule):
#     def __init__(self,
#                  ddconfig,
#                  lossconfig,
#                 #  n_embed,
#                 #  embed_dim,
#                  ckpt_path=None,
#                  ignore_keys=[],
#                  multimodal_ckpt_dict={}
#                  ):
#         pass

#     def instantiate_multimodels(ckpt_dict):
#         for name, ckpt in ckpt_dict.items():
            