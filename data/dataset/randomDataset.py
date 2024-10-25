import torch
from torch.utils.data import Dataset

class RandomImageDataset(Dataset):
    def __init__(self, input_size=(1, 1024, 1024), num_samples=100, label_value=1):
        """
        初始化数据集
        :param input_size: 每张图像的尺寸
        :param num_samples: 数据集中样本数
        :param label_value: 标签值 (例如，全 1)
        """
        self.input_size = input_size
        self.num_samples = num_samples
        self.label_value = label_value

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(tuple(self.input_size))  # 生成随机图像
        label = torch.ones(tuple(self.input_size)) * self.label_value  # 生成标签
        return image, label
