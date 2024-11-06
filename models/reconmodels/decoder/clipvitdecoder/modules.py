import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pytorch_lightning as pl

class ImageDownsample(nn.Module):
    def __init__(self, in_size, scale_factor=None):
        """
        Args:
            out_size (int or tuple)
            scale_factor (float or int, optional)
        """
        super().__init__()
        self.in_size = in_size
        self.scale_factor = scale_factor
        if self.scale_factor is not None:
            self.out_size = int(in_size * self.scale_factor)
        else:
            self.out_size = self.in_size
        assert isinstance(self.out_size, int), "Output size must be an integer"

    def encode(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def decode(self, x):
        return nn.functional.interpolate(x, size=self.in_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        assert len(x.shape) == 4, "Input tensor must have 4 dimensions, it is an image tensor"
        x = self.encode(x)
        x = self.decode(x)
        return x
