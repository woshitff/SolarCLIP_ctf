import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import scipy.linalg


class FID:
    def __init__(self):
        self.transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

        self.model = models.inception_v3(pretrained=True, transform_input=False)
        self.model.eval()

    def get_features(self, inputs, recons):
        inputs = self.transform(inputs)
        recons = self.transform(recons)

        with torch.no_grad():
            inputs_features = self.model(inputs)
            recons_features = self.model(recons)
        return inputs_features.cpu().numpy(), recons_features.cpu().numpy() 
    
    def calculate_fid(self, inputs, recons):
        inputs_features, recons_features = self.get_features(inputs, recons)

        mu_inputs = np.mean(inputs_features, axis=0)
        sigma_inputs = np.cov(inputs_features, rowvar=False)

        mu_recons = np.mean(recons_features, axis=0)
        sigma_recons = np.cov(recons_features, rowvar=False)

        diff = mu_inputs - mu_recons
        diff_squared = np.sum(diff**2)
        covmean, _ = scipy.linalg.sqrtm(sigma_inputs.dot(sigma_recons), disp=False)
        fid = diff_squared + np.trace(sigma_inputs + sigma_recons - 2 * covmean)

        print(f"FID score between original and reconstructed images: {fid}")
        return fid
    

class InceptionScore:
    def __init__(self):
        pass


class PSNR:
    def __init__(self):
        pass

    def calculate_psnr(self, inputs, recons):
        mse = F.mse_loss(inputs, recons)
        max_pixel_value = np.max(inputs.cpu().numpy())
        psnr = 10 * torch.log10((max_pixel_value ** 2) / mse)
        
        return psnr.item()
    
class PSIM:
    pass

class SSIM:
    pass 
