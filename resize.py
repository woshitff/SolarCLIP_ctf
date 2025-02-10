import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# 读取图片
image_path = "/mnt/tianwen-tianqing-nas/tianwen/ctf/solarclip/zyz_106/logs/reconmodels/autoencoder/vae/ae0131_first/images/val/inputs_gs-066924_e-000233_b-000000.png"  # 你的图片路径
original_img = Image.open(image_path).convert("RGB")

# 获取原始尺寸
orig_width, orig_height = original_img.size

# 计算缩小后的尺寸（下采样 36 倍）
small_width, small_height = int(orig_width/np.sqrt(8)), int(orig_height /np.sqrt(8))

# 转换为张量
transform_to_tensor = transforms.ToTensor()
img_tensor = transform_to_tensor(original_img).unsqueeze(0)  # 增加 batch 维度

# 使用双线性插值进行下采样
downsampled_img = torch.nn.functional.interpolate(img_tensor, size=(small_height, small_width), mode="bilinear", align_corners=False)

# 再上采样回原尺寸
upsampled_img = torch.nn.functional.interpolate(downsampled_img, size=(orig_height, orig_width), mode="bilinear", align_corners=False)

# 转回 PIL Image
transform_to_pil = transforms.ToPILImage()
upsampled_pil = transform_to_pil(upsampled_img.squeeze(0))  # 移除 batch 维度

# 保存原始图片
# original_img.save("original.png")

# 保存恢复后的图片
upsampled_pil.save("recovered.png")

print("原始图片和恢复后的图片已保存！")
