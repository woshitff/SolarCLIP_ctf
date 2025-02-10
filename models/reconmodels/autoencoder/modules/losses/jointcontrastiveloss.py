import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class JointContrastiveLoss(nn.Module):
    def __init__(self,
                 models: dict = None
                 ):
         super().__init__()
         self.models = models
      
    def forward(self, 
                data):

        features = {}
        for i, (modal_name, model) in enumerate(self.models.items()):
            feature = model.encode(data[:, i, :, :, :])
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