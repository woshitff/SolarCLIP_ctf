import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange

def JointContrastiveLoss(models: dict = None, data: torch.Tensor = None, modal: str = None, mode:str = 'feature', **kwargs):
    """
    Joint Contrastive Loss for multi-modal data.
    Args:
        models (dict): Dictionary of models for each modality.
        data (torch.Tensor): Input data of shape (batch_size, num_modalities, channels, height, width).
        modal (str): The name of the target modality for contrastive loss.
        mode (str): The mode of the loss. 'feature' or 'logit'.
    Returns:
        loss (torch.Tensor): The computed contrastive loss.
    """
    moments = {}
    for i, (modal_name, model) in enumerate(models.items()):
        if isinstance(model, DDP):
            feature, _ = model.module.encode(data[:, i, :, :, :])
        else:
            feature, _ = model.encode(data[:, i, :, :, :])
        # print(f"feature shape: {feature.shape}")
        if mode == 'feature':
            feature = rearrange(feature, 'b c h w -> b h w c')
            feature= feature / (feature.norm(dim=-1, keepdim=True)+1e-32)
            moments.update({f"{modal_name}":feature})
        elif mode == 'logit':
            logit = model.classify(feature) # (b, c)
            logit = logit / (logit.norm(dim=-1, keepdim=True)+1e-32) # Normalize logits
            moments.update({f"{modal_name}":logit})
        else:
            raise ValueError("Invalid mode. Choose 'feature' or 'logit'.")

    target_moment = moments[modal]  
    other_moment = {name: moment for name, moment in moments.items() if name != modal}
    if mode == 'feature':
        loss = sum((target_moment * moment).sum(dim=-1).mean() for moment in other_moment.values()) / len(other_moment)
    elif mode == 'logit':
        loss = 0
        for name, moment in other_moment.items():
            loss += F.mse_loss(target_moment, moment) / len(other_moment)

    return loss