import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class JointContrastiveLoss(nn.Module):
    def __init__(self,
                #  basic_modal_key,
                 ):
        pass
        # self.basic_modal_key = basic_modal_key

    # def calculate_contrastloss(base_modal, paired_modal):
    #     """
    #     base_modal: ususally HMI modal, shape (B, C, H, W)
    #     paired_modal: ususally a single modal using contrastive-learning-like way to learn a better representation, shape (B, C, H, W)
    #     """
    #     base_features = rearrange(base_modal, 'b c h w -> b h w c')
    #     paired_features = rearrange(paired_modal, 'b c h w -> b h w c')

    #     base_features = base_features / (base_features.norm(dim=-1, keepdim=True)+1e-32)
    #     paired_features = paired_features / (paired_features.norm(dim=-1, keepdim=True)+1e-32)
    def forward(self, 
                split="train", 
                **modals):
        print(f"Received {len(modals)} modals: {list(modals.keys())}")

        features = {}
        for name, modal in modals.items():
            feature = rearrange(modal, 'b c h w -> b h w c')
            feature= feature / (feature.norm(dim=-1, keepdim=True)+1e-32)
            features[name] = feature

        inner_prod = None
        for name, feature in features.items():
            if inner_prod is None:
                inner_prod = feature
            else:
                inner_prod = inner_prod * feature
        inner_prod = inner_prod.sum(dim=-1)

        loss = torch.mean(inner_prod)
        log = {
            "{}/total_contrastive_loss".format(split): loss.clone().detach().mean()
        }

        return loss, log