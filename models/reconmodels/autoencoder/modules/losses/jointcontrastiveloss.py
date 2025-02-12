import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange

def JointContrastiveLoss(models: dict = None, data: torch.Tensor = None):

        features = {}
        for i, (modal_name, model) in enumerate(models.items()):
            if isinstance(model, DDP):
                feature, _ = model.module.encode(data[:, i, :, :, :])
            else:
                feature, _ = model.encode(data[:, i, :, :, :])
            # print(f"feature shape: {feature.shape}")
            feature = rearrange(feature, 'b c h w -> b h w c')
            feature= feature / (feature.norm(dim=-1, keepdim=True)+1e-32)
            features.update({f"{modal_name}":feature})

        inner_prod = None
        for name, feature in features.items():
            if inner_prod is None:
                inner_prod = feature
            else:
                inner_prod = inner_prod * feature
        inner_prod = inner_prod.sum(dim=-1)

        loss = torch.mean(inner_prod)
    
        return loss