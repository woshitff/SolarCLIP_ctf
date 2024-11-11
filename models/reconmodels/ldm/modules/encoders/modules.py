import torch.nn as nn
from einops import rearrange

from models.clipmodels.modules.vit import Remove_class_token
from models.reconmodels.ldm.util import instantiate_from_config


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class ReshapeTo2D(nn.Module):
    def __init__(self, h=16, w=16):
        super().__init__()
        self.h = h
        self.w = w

    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h=self.h, w=self.w)
    
class LinearProjectionToImage(nn.Module):
    def __init__(self, input_dim=(768, 16, 16), output_dim=(3, 64, 64)):
        super(LinearProjectionToImage, self).__init__()
        input_size = input_dim[0] * input_dim[1] * input_dim[2]  # 768 * 16 * 16
        output_size = output_dim[0] * output_dim[1] * output_dim[2]  # 3 * 64 * 64
        self.fc = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  
        x = self.fc(x)             
        x = self.activation(x)      
        return x.reshape(x.size(0), 3, 64, 64)  

#-*************************************-#

class SolarCLIPImageEmbedder(nn.Module):
    """Get image embedding from SolarCLIP and project it to image space."""
    def __init__(self, 
                 solarclip_config,
                 embedding_key='magnet_image', 
                 projection_type='Linear', 
                 **kwargs):
        super().__init__()
        self.projection_type = projection_type
        self.embedding_key = embedding_key

        self._init_solarclip(solarclip_config)
        self.ReshapeProjection = self._get_projection(projection_type)

    def _init_solarclip(self, solarclip_config, freeze=True):
        solarclip = instantiate_from_config(solarclip_config)
        if freeze:
            self.solarclip = solarclip.eval()
            self.solarclip.train = disabled_train
            for param in self.solarclip.parameters():
                param.requires_grad = False
        if self.embedding_key == 'magnet_image':
            self.solarclip = self.solarclip.visual_hmi
        elif self.embedding_key == 'aia0094_image':
            self.solarclip = self.solarclip.visual_aia
        else:
            raise ValueError(f"Unknown embedding key {self.embedding_key}")
        
    def _get_projection(self, projection_type):
        projectin_options = {
            "ConvTrans": nn.Sequential(
                ReshapeTo2D(16, 16),
                nn.ConvTranspose2d(768, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(16, 128),
                nn.Tanh(),
                nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(1, 3),
                nn.Tanh()
                ),
            "Linear": nn.Sequential(
                ReshapeTo2D(16, 16),
                LinearProjectionToImage(input_dim=(768, 16, 16), output_dim=(3, 64, 64))
                )
        }

        if projection_type in projectin_options:
            return projectin_options[projection_type]
        else:
            raise ValueError(f"Unknown projection type {projection_type}")

    def forward(self, x):
        x = self.solarclip(x)
        x = Remove_class_token()(x) # (B, 257, 768) -> (B, 256, 768)
        x = self.ReshapeProjection(x)
        return x
        
