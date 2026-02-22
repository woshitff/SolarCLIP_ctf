import numpy as np
import tqdm

import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from omegaconf import OmegaConf

from data.dataset.SolarDataset_v2 import multimodal_dataset
from data.utils import load_list


def transfer_id_to_date_V2(date_id,y_start=2010, m_start = 5, d_start = 1, y_end=2024, m_end = 7, d_end = 1): # todo
    start_date = datetime(y_start, m_start, d_start)
    end_date = datetime(y_end, m_end, d_end)
    if date_id < 0 or date_id > (end_date - start_date).days*24*60:
        raise ValueError(f'date_id out of range, should be in 0 to {(end_date - start_date).days*24*60}')
    current_date = start_date + timedelta(days=date_id)

    return current_date

def transfer_date_to_id_V2(date, y_start=2010, m_start = 5, d_start = 1, y_end=2024, m_end = 7, d_end = 1): # todo
    start_date = datetime(y_start, m_start, d_start)
    end_date = datetime(y_end, m_end, d_end)
    if date < start_date or date > end_date:
        raise ValueError(f'date out of range, should be in {start_date} to {end_date}')
    date_id = (date - start_date).days*24*60
    return date_id


class ClassificationDataset(Dataset):
    def __init__(self, solardataset: multimodal_dataset, time_interval = [[2010,5,1], [2010,12,24]], label_type: str='ABCMX', label_dir: str='SolarCLIP_ctf/downstreams/flare_prediction/data/flare_label_dict.npy', start_date = [2010,5,1,1,34], end_date = [2017,6,27,21,19]):
        super().__init__()

        if datetime(*time_interval[0]) < datetime(*start_date) or datetime(*time_interval[1]) > datetime(*end_date):
            raise ValueError(f'The date range in yaml file should be within the range of flare data from {start_date} to {end_date}')
        
        self.solar_data = solardataset
        print(self.solar_data.exist_idx)
        start_id = transfer_date_to_id_V2(datetime(*time_interval[0]))
        end_id = transfer_date_to_id_V2(datetime(*time_interval[1]))
        self.solar_data.exist_idx = self.solar_data.exist_idx[(self.solar_data.exist_idx >= start_id) & (self.solar_data.exist_idx <= end_id)] # filter the data to the time interval

        self.label_dict =  np.load(label_dir, allow_pickle=True).item()

        if label_type == 'ABCMX':
            pass
        elif label_type == 'CM':
            rearrange = [0,0,0,1,2,2] # [0,A,B] = 0, [C] = 1, [M,X] = 2
            for key in self.label_dict.keys():
                self.label_dict[key] = rearrange(self.label_dict[key])
        elif label_type == 'CX':
            rearrange = [0,0,0,1,1,2] # [0,A,B] = 0, [C,M] = 1, [X] = 2
            for key in self.label_dict.keys():
                self.label_dict[key] = rearrange(self.label_dict[key])
        elif label_type == '>=C':
            rearrange = [0,0,0,1,1,1] # [0,A,B] = 0, [C,M,X] = 1
            for key in self.label_dict.keys():
                self.label_dict[key] = rearrange(self.label_dict[key])
        elif label_type == '>=M':
            rearrange = [0,0,0,0,1,1] # [0,A,B,C] = 0, [M,X] = 1
            for key in self.label_dict.keys():
                self.label_dict[key] = rearrange(self.label_dict[key])
        elif label_type == '>=X':
            rearrange = [0,0,0,0,0,1] # [0,A,B,C,M] = 0, [X] = 1
            for key in self.label_dict.keys():
                self.label_dict[key] = rearrange(self.label_dict[key])
        else:
            raise ValueError('label_type should be ABCMX, CM or CX or >=C or >=M or >=X')
        
        self.remove_excessive_0_class()
    
    def remove_excessive_0_class(self, zero_other_ratio = 1.0):
        # zero_other_ratio = len(label_0_idxs) / len(label_A_idxs) + len(label_B_idxs) + len(label_C_idxs) + len(label_M_idxs) + len(label_X_idxs)

        # record different labels' indexes
        label_0_idxs = []
        label_A_idxs = []
        label_B_idxs = []
        label_C_idxs = []
        label_M_idxs = []
        label_X_idxs = []
        label_cache = [label_0_idxs, label_A_idxs, label_B_idxs, label_C_idxs, label_M_idxs, label_X_idxs]
        for idx in range(len(self.solar_data.exist_idx)):
            label_date = transfer_id_to_date_V2(idx)
            label_date = int(label_date.strftime('%Y%m%d'))
            label = self.label_dict[label_date]
            label_cache[label].append(idx)
        label_0_idxs = np.array(label_0_idxs)
        label_A_idxs = np.array(label_A_idxs)
        label_B_idxs = np.array(label_B_idxs)
        label_C_idxs = np.array(label_C_idxs)
        label_M_idxs = np.array(label_M_idxs)
        label_X_idxs = np.array(label_X_idxs)

        # remove excessive 0 class
        len_others = len(label_A_idxs) + len(label_B_idxs) + len(label_C_idxs) + len(label_M_idxs) + len(label_X_idxs)
        len_0 = len(label_0_idxs)
        if len_0 > len_others * zero_other_ratio:
            label_0_idxs = label_0_idxs[::int(len_0/(len_others * zero_other_ratio))]
        
        # update solar_data
        new_idxs = np.concatenate([label_0_idxs, label_A_idxs, label_B_idxs, label_C_idxs, label_M_idxs, label_X_idxs])
        new_idxs = np.sort(new_idxs) # time order
        self.solar_data.exist_idx = new_idxs

        # get focal weight
        focal_weight = np.array([len(label_0_idxs), len(label_A_idxs), len(label_B_idxs), len(label_C_idxs), len(label_M_idxs), len(label_X_idxs)])
        focal_weight = focal_weight/np.sum(focal_weight)
        focal_weight = 1- focal_weight
        self.focal_weight = focal_weight/np.sum(focal_weight)

        print(f' mode {self.mode}: 0 class has {len(label_0_idxs)} samples, A class has {len(label_A_idxs)} samples, B class has {len(label_B_idxs)} samples, C class has {len(label_C_idxs)} samples, M class has {len(label_M_idxs)} samples, X class has {len(label_X_idxs)} samples')
        print(f'focal weight: {self.focal_weight}')


    def __len__(self):
        return len(self.solar_data.exist_idx)

    def __getitem__(self, idx):
        # get T+1 frame, remove last frame and take the last label
        image = self.solar_data[idx] # b, modal, c, h, w
        date = transfer_id_to_date_V2(self.solar_data.exist_idx[idx])
        # date = date + timedelta(hours=1) # the flare data is hourly, so we need to add 1 hour to match the flare data
        date = int(date.strftime('%Y%m%d'))
        label = self.label_dict[date]
        label = torch.tensor(label)
        return image, label, self.focal_weight[label]
    
if __name__ == '__main__':
    from .utils import instantiate_from_config
    config = OmegaConf.load('./downstreams/flare_prediction/config.yaml')
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    train_dataset = data.datasets["train"]
    classification_dataset = ClassificationDataset(train_dataset, time_interval=[[2010,5,1], [2010,12,24]], label_type='ABCMX', label_dir='SolarCLIP_ctf/downstreams/flare_prediction/data/flare_label_dict.npy')