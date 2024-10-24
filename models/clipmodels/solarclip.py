from typing import Tuple, Union

import torch
import torch.nn as nn
import numpy as np

from models.clipmodels.modules.vit import VisionTransformer

class SolarCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # mag vision
                 image_resolution_hmi: int,
                 vision_layers_hmi: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size_hmi: int,
                 # 11 channels
                 image_resolution_aia: int,
                 vision_layers_aia: Union[Tuple[int, int, int, int], int],
                 #vision_width: int,
                 vision_patch_size_aia: int,
                 # loss token type
                 transformer_token_type: str,
                 norm_type: str 
                 ):
        super().__init__()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.transformer_token_type = transformer_token_type

        vision_heads = vision_width // 64
        self.visual_hmi = VisionTransformer(
                in_channels=1,
                input_resolution=image_resolution_hmi,
                patch_size=vision_patch_size_hmi,
                width=vision_width,
                layers=vision_layers_hmi,
                heads=vision_heads,
                output_dim=embed_dim,
                token_type = transformer_token_type,
                norm_type= norm_type
            )
        
        self.visual_aia = VisionTransformer(
                in_channels=1,
                input_resolution=image_resolution_aia,
                patch_size=vision_patch_size_aia,
                width=vision_width,
                layers=vision_layers_aia,
                heads=vision_heads,
                output_dim=embed_dim,
                token_type = transformer_token_type,
                norm_type = norm_type
            )

    @property
    def dtype(self):
        # return self.visual_hmi.conv1.weight.dtype
        return self.visual_hmi.conv1.conv1.weight.dtype

    def encode_hmi(self, image_hmi):
        return self.visual_hmi(image_hmi.type(self.dtype))

    def encode_aia(self, image_aia):
        return self.visual_aia(image_aia.type(self.dtype))

    def forward(self, image_hmi, image_aia, token_weight_1=None, token_weight_2=None):

        hmi_features = self.encode_hmi(image_hmi)   #shape = [batch_size, length,embed_dim]
        aia_features = self.encode_aia(image_aia)
        
        # normalized features
        hmi_features = hmi_features / (hmi_features.norm(dim=-1, keepdim=True)+1e-32)
        aia_features = aia_features / (aia_features.norm(dim=-1, keepdim=True)+1e-32)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()

        if self.transformer_token_type == 'class embedding':
            logits_per_hmi = logit_scale * hmi_features @ aia_features.t()
            logits_per_aia = logits_per_hmi.t()
            inner_cor_matrix = None

        elif self.transformer_token_type == 'all embedding':
            B, L = hmi_features.shape[0], hmi_features.shape[1]
            if token_weight_1 is None:
                token_weight_1 = torch.ones([B,L],dtype = hmi_features.dtype,device = hmi_features.device)
            if token_weight_2 is None:
                token_weight_2 = torch.ones([B,L],dtype = aia_features.dtype,device = aia_features.device)
            assert (token_weight_1.shape == (B, L) and token_weight_2.shape == (B, L)) # [B,L] tensor
            token_weight_1 = token_weight_1.unsqueeze(-1)
            token_weight_2 = token_weight_2.unsqueeze(-1)

            hmi_features = torch.einsum('BLD,BLd->BLD', hmi_features, token_weight_1)
            aia_features = torch.einsum('BLD,BLd->BLD', aia_features, token_weight_2)
            inner_cor_matrix = torch.einsum('BLD,BlD->BLl', hmi_features, aia_features)
            inner_cor_matrix = inner_cor_matrix.mean(dim=0) # [L,L]
            cor_matrix = torch.einsum('BLD,bLD->BbL', hmi_features, aia_features)
            cor_matrix = cor_matrix.mean(dim=-1) # [B,B]

            logits_per_hmi = logit_scale * cor_matrix
            logits_per_aia = logits_per_hmi.t()
            inner_cor_matrix = logit_scale * inner_cor_matrix

        # shape = [global_batch_size, global_batch_size]
        # return logits_per_hmi, logits_per_H
        return logits_per_hmi, logits_per_aia, inner_cor_matrix
    
    def calculate_loss(self, hmi_image, aia_image, inner_loss_rate = 0, token_weight_1 = None, token_weight_2 = None, criterion = torch.nn.functional.cross_entropy):

        logits_per_hmi, logits_per_aia, inner_cor_matrix = self.forward(hmi_image, aia_image, token_weight_1, token_weight_2)
        ground_truth = torch.arange(len(hmi_image), dtype=torch.long, device=hmi_image.device)
        
        loss_hmi = criterion(logits_per_hmi, ground_truth)
        loss_aia = criterion(logits_per_aia, ground_truth)
        loss = (loss_hmi + loss_aia) / 2
        acc = (torch.argmax(logits_per_hmi, dim=1) == ground_truth).float().mean().item()

        assert inner_loss_rate >=0
        if inner_loss_rate > 0:
            ground_truth = torch.arange(inner_cor_matrix.shape[-1], dtype=torch.long, device=inner_cor_matrix.device)#[l]
            # ground_truth = ground_truth.unsqueeze(0).expand(inner_cor_matrix.shape[0],-1) # [L,L]
            loss_inner = criterion(inner_cor_matrix,ground_truth)/2
            loss_inner = loss_inner + criterion(inner_cor_matrix.t(),ground_truth)/2
        else:
            loss_inner = torch.tensor(0, dtype=torch.float32, device=inner_cor_matrix.device)

        return loss, loss_inner, acc, logits_per_hmi, inner_cor_matrix