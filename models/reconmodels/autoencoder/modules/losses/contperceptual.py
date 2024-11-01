import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

class LPIPS(nn.Module):
    # Learned perceptual image patch similarity (LPIPS) loss
    # see https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/losses/contperceptual.py
    def __init__(self, 
                 rec_loss_type='l2',
                 log_var_init=0.0,
                 kl_weight=0.1,
                 perceptual_weight=1):
        super().__init__()
        self.rec_loss_type = rec_loss_type
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight

        self.logvar = nn.Parameter(torch.ones(size=())*log_var_init)
        self.perceptual_loss = lpips.LPIPS(net='vgg').eval()

        
    def forward(self, inputs, recons, posteriors, weights=None, log_split="train"):

        if self.rec_loss_type == 'l1':
            rec_loss = torch.abs(inputs.contiguous() - recons.contiguous())
        elif self.rec_loss_type == 'l2':
            rec_loss = F.mse_loss(inputs.contiguous(), recons.contiguous(), reduction='none')
        else:
            raise ValueError(f"Invalid rec_loss_type: {self.rec_loss_type}")
        
        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(inputs.repeat(1, 3, 1, 1).contiguous(), recons.repeat(1, 3, 1, 1).contiguous())
            rec_loss = rec_loss + self.perceptual_weight * perceptual_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        if isinstance(posteriors, tuple):
            post_mu, post_logvar = posteriors
            kl_loss = -0.5 * torch.sum(1 + post_logvar - post_mu.pow(2) - post_logvar.exp())
            kl_loss = kl_loss / post_mu.shape[0]
        else:
            kl_loss = torch.tensor(0.0)
        
        loss = weighted_nll_loss + self.kl_weight * kl_loss

        loss_dict = {
            "{}/total_loss".format(log_split): loss.clone().detach().item(),
            "{}/rec_loss".format(log_split): rec_loss.mean().clone().detach().item(),
            "{}/nll_loss".format(log_split): nll_loss.clone().detach().item(),
            "{}/kl_loss".format(log_split): kl_loss.clone().detach().item(),
        }

        return loss, loss_dict