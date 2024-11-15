import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from models.reconmodels.autoencoder.models.vqvae.modules.taming_vqgan.discriminator.model import NLayerDiscriminator, weights_init
from models.reconmodels.autoencoder.models.vqvae.modules.taming_vqgan.losses.lpips import LPIPS
from models.reconmodels.autoencoder.models.vqvae.modules.taming_vqgan.losses.vqperceptual import hinge_d_loss, vanilla_d_loss


def hinge_d_loss_with_exemplar_weights(logits_real, logits_fake, weights):
    assert weights.shape[0] == logits_real.shape[0] == logits_fake.shape[0]
    loss_real = torch.mean(F.relu(1. - logits_real), dim=[1,2,3])
    loss_fake = torch.mean(F.relu(1. + logits_fake), dim=[1,2,3])
    loss_real = (weights * loss_real).sum() / weights.sum()
    loss_fake = (weights * loss_fake).sum() / weights.sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use

def exists(val):
    return val is not None

def l1(x, y):
    return torch.abs(x-y)

def l2(x, y):
    return torch.pow((x-y), 2)


class VQDoubleLoss(nn.Module):
    def __init__(self, codebook_weight=1.0, pixelloss_weight=1.0, classifier_weight=1.0,
                 n_classes=None,
                 pixel_loss="l1"):
        super().__init__()
        assert pixel_loss in ["l1", "l2"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.classifier_weight = classifier_weight

        if pixel_loss == "l1":
            self.pixel_loss = l1
        else:
            self.pixel_loss = l2

        self.n_classes = n_classes

    def forward(self, codebook_loss, inputs, reconstructions,
                ind_first, logits,
                split="train", predicted_indices=None):
        if not exists(codebook_loss):
            codebook_loss = torch.tensor([0.]).to(inputs.device)
        rec_loss = self.pixel_loss(inputs.contiguous(), reconstructions.contiguous())

        # ind_first shape (B*H*W, )
        # logits shape (B*H*W, n_classes)
        # print(f"ind_first shape: {ind_first.shape}, logits shape: {logits.shape}")
        if self.classifier_weight > 0:
            classifier_loss = F.cross_entropy(logits, ind_first)
            rec_loss = rec_loss + self.classifier_weight * classifier_loss
        else:
            classifier_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        loss = nll_loss + self.codebook_weight * codebook_loss.mean()
        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/classifier_loss".format(split): classifier_loss.detach().mean(),
                # "{}/d_weight".format(split): d_weight.detach(),
                # "{}/disc_factor".format(split): torch.tensor(disc_factor),
                # "{}/g_loss".format(split): g_loss.detach().mean(),
                }
        if predicted_indices is not None:
            assert self.n_classes is not None
            with torch.no_grad():
                perplexity, cluster_usage = measure_perplexity(predicted_indices, self.n_classes)
            log[f"{split}/perplexity"] = perplexity
            log[f"{split}/cluster_usage"] = cluster_usage
        return loss, log

        