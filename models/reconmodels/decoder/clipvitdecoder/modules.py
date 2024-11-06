import torch
import torch.nn as nn
import torchvision.transforms as transforms

class ImageResizeTarget(nn.Module):
    def __init__(self, out_size, scale_factor=None):
        """
        Args:
            out_size (int or tuple)
            scale_factor (float or int, optional)
        """
        super(ImageResizeTarget, self).__init__()
        self.out_size = out_size
        self.scale_factor = scale_factor
    
    def forward(self, batch, k):
        if k == 'magnet_image':
            x = batch[:, 0, :, :, :]
        elif k == 'aia0094_image':
            x = batch[:, 1, :, :, :]
        else:
            raise NotImplementedError(f"Key {k} not supported")
        
        if len(x.shape) == 3:
            x = x[..., None]
        
        if self.scale_factor is not None:
            new_height = int(x.shape[2] // self.scale_factor)
            new_width = int(x.shape[3] // self.scale_factor)
            target_size = (new_height, new_width)
        else:
            target_size = self.out_size

        resize_transform = transforms.Resize(target_size)
        x = resize_transform(x)
        
        x = x.to(memory_format=torch.contiguous_format).float()

        return x
