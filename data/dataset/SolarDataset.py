import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..utils import read_pt_image as read_image
from ..utils import load_list, get_modal_dir

import random
from tqdm import tqdm

     
def transfer_log1p(input_array):
    if isinstance(input_array,np.ndarray):
        return np.sign(input_array)*np.log1p(np.abs(input_array))
    elif isinstance(input_array,torch.Tensor):
        return torch.sign(input_array)*torch.log1p(torch.abs(input_array))
    else:
        raise ValueError('input_array should be numpy array or torch tensor')

def image_preprocess(image_list, image_size = 224, p_flip = 0.5, p_rotate = 90):
    N = len(image_list)
    channels = np.zeros(N, dtype=int)
    for i in range(N):
        image = image_list[i]
        if isinstance(image, np.ndarray):
            image = np.nan_to_num(image, nan=0.0)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(image_size),
            ])
            image = transform(image)
            image_list[i] = image
        elif isinstance(image, torch.Tensor):
            image = image.unsqueeze(0)
            transform = transforms.Compose([
                transforms.Resize(image_size),
            ])
            image = transform(image)
            image_list[i] = image
        else:
            raise ValueError('image should be numpy array or torch tensor')
        channels[i]=image.shape[0]
    
    image = torch.cat(image_list, dim=0)
    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p_flip),
            transforms.RandomVerticalFlip(p_flip),
            transforms.RandomRotation(p_rotate),
        ]) 
    image = transform(image)

    for i in range(N):
        image_list[i] = image[channels[:i].sum():channels[:i+1].sum()]

    image = torch.stack(image_list, dim=0)
    return image

def enhance_funciton(image, enhance_type = 'log1p', rescale_value = 1):
    
    if enhance_type == 'log1p':
        image = transfer_log1p(image)
    elif enhance_type == 'None':
        pass
    else:
        raise ValueError('enhance_type should be log1p or None')
    
    image = image*rescale_value
    return image

# todo for current version, one modal dataset only load exist_idx, multi-modal dataset can choose whether load images.
class Dataset_one_modal(Dataset):
    def __init__(self, modal, exist_idx: np.ndarray):

        self.modal = modal
        self.exist_idx = exist_idx

    def __len__(self):
        return len(self.exist_idx)

    def load_images(self, idx_list):
        image_list = {}
        for idx in tqdm(idx_list, desc=f'Loading {self.modal} images'):
            if self.exist_idx[idx] == False:
                continue
            idx = int(idx)
            path = get_modal_dir(self.modal, idx)[1]
            img = read_image(path)
            image_list[idx] = img
        return image_list
    
    def __getitem__(self, idx):
        idx = int(idx)
        img_path = get_modal_dir(self.modal, idx)[1]
        is_exist = self.exist_idx[idx]
        return img_path, is_exist

# todo version 1.0, 20240813, if load exist_idx, get image from one-modal-dataset, if load images, save images and del one-modal-dataset
# todo version 1.1, 20240813, move filter exist_idx to independent function, call it during initialization
class multimodal_dataset(Dataset):
    def __init__(self, modal_list = ['magnet','0094'], load_imgs = False, enhance_list = [224,0.5,90], time_interval = [0,7452000],time_step = 1): #time_step = 1 means get every data
        # 定义数据集
        self.dataset = [] 
        time_slice = slice(time_interval[0],time_interval[1],time_step)
        if 'magnet' in modal_list:
            # mag_dir_list = load_list('./Data/dir_list/magnet_dir_list_pt.pkl')[time_slice]
            mag_idx = load_list('./data/idx_list/magnet_exist_idx.pkl')
            self.dataset.append(Dataset_one_modal('magnet',mag_idx))

        if '0094' in modal_list:
            # h0094_dir_list = load_list('./Data/dir_list/0094_dir_list_pt.pkl')[time_slice]
            h0094_idx = load_list('./data/idx_list/0094_exist_idx.pkl')
            self.dataset.append(Dataset_one_modal('0094',h0094_idx))

        # find the all exist index
        exist_list = []
        for i in range(len(self.dataset)):
            dataset = self.dataset[i]
            exist_list.append(dataset.exist_idx)
            print(f' {modal_list[i]} has {np.sum(exist_list[i])} samples')

        # first test, if all modal has the same exist data
        self.exist_idx = self.filter_exist_idx(exist_list, time_interval, time_step)
        print(f'All modal has {len(self.exist_idx)} samples')

        self.enhance_list = enhance_list

        self.load_imgs = load_imgs
        if load_imgs:
            list_of_image_dic = []
            for i in range(len(self.dataset)):
                list_of_image_dic.append(self.dataset[i].load_images(self.exist_idx))
            del self.dataset
            self.image_dic = {}
            for idx in tqdm(self.exist_idx, desc='Merging images'):
                self.image_dic[idx] = image_preprocess([image_dic[idx] for image_dic in list_of_image_dic], image_size = enhance_list[0], p_flip = enhance_list[1], p_rotate = enhance_list[2])
                for image_dic in list_of_image_dic:
                    del image_dic[idx]
            del list_of_image_dic

    def filter_exist_idx(self,exist_list, time_interval, time_step):
        exist_list = np.array(exist_list)
        exist_idx = np.all(exist_list,axis=0)
        exist_idx = np.nonzero(exist_idx)[0] # get the index of all exist data
        # second test, if the idx follows the time interval
        exist_idx = exist_idx[exist_idx>=time_interval[0]]
        exist_idx = exist_idx[exist_idx<time_interval[1]]
        exist_idx = exist_idx[exist_idx%time_step==0]
        # exist_idx = exist_idx[::time_step]

        return exist_idx

    def __len__(self):
        return len(self.exist_idx)
    
    def __getitem__(self, idx):
        if self.load_imgs:
            return self.image_dic[self.exist_idx[idx]]
        else:
            image_list = []
            for i in range(len(self.dataset)):
                path,_ = self.dataset[i][self.exist_idx[idx]]
                img = read_image(path)

                image_list.append(img)
            image = image_preprocess(image_list, image_size = self.enhance_list[0], p_flip = self.enhance_list[1], p_rotate = self.enhance_list[2])
            image = enhance_funciton(image, enhance_type = 'log1p', rescale_value = 1)
            return image

if __name__ == '__main__':

    dataset = multimodal_dataset(modal_list=['magnet','0094'], load_imgs=True, enhance_list=[224,0.5,90],time_interval=[0,7452000],time_step=60)
    # print(len(dataset))
    # start_date = transfer_date_to_id(2010, 5, 1)
    # end_date = transfer_date_to_id(2020, 6, 30)
    # time_interval = [start_date, end_date]

    # start_date = transfer_date_to_id(2020, 6, 30)
    # end_date = transfer_date_to_id(2024, 6, 30)
    # time_interval = [start_date, end_date]
    # dataset = multimodal_dataset(modal_list, load_imgs, enhance_list, time_interval,time_step)