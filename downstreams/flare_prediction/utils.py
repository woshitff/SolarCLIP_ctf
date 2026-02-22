import pandas as pd
import numpy as np
import torch
import wget
import os
import requests

from datetime import datetime, timedelta
from astropy.io import fits
import importlib

import torch

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def transfer_original_into_dict(csv_path, start_date = [2010,5,1,1,34], end_date = [2017,6,27,21,19]):
    data = pd.read_csv(csv_path)

    # 0 no flare
    # 1 A
    # 2 B
    # 3 C
    # 4 M
    # 5 X
    def transfer_class(x):
        if 'A' in x:
            return 1
        elif 'B' in x:
            return 2
        elif 'C' in x:
            return 3
        elif 'M' in x:
            return 4
        elif 'X' in x:
            return 5
        else:
            raise ValueError('Invalid class', x)

    # initialize the data_dict
    if start_date[4] ==0:
        current_date_ = datetime(start_date[0], start_date[1], start_date[2], start_date[3])
    else:
        current_date_ = datetime(start_date[0], start_date[1], start_date[2], start_date[3]) - timedelta(hours=1)
    if end_date[4] ==0:  
        end_date_ = datetime(end_date[0], end_date[1], end_date[2], end_date[3], end_date[4])
    else:
        end_date_ = datetime(end_date[0], end_date[1], end_date[2], end_date[3], end_date[4]) + timedelta(hours=1)
    label_dict = {}
    while current_date_ <= end_date_:
        label_dict[int(current_date_.strftime('%Y%m%d%H%M'))] = 0
        current_date_ += timedelta(hours=1)
    
    for i in range(len(data)):
        day = data['Date'][i]
        start = data['Start'][i]
        end = data['End'][i]
        print(len(data), i, day, start, end)
        class_ = transfer_class(data['Class'][i])

        start = np.floor(start/100) # remove minute, only consider hour
        end = np.ceil(end/100)
        start_date_ = datetime.strptime(str(day), '%Y%m%d')
        current_date_ = start_date_ + timedelta(hours=start)
        end_date_ = start_date_ + timedelta(hours=end)
        while current_date_ <= end_date_:
            date = int(current_date_.strftime('%Y%m%d%H%M'))
            label_dict[date] = max(label_dict[date], class_)
            current_date_ += timedelta(hours=1)

    return label_dict

def merge_hourly_labels(label_dict, start_date = [2010,5,1,1,34], end_date = [2017,6,27,21,19]):
    # merge the hourly labels inot daily labels by taking the maximum value

    if start_date[4] ==0:
        current_date_ = datetime(start_date[0], start_date[1], start_date[2], start_date[3])
    else:
        current_date_ = datetime(start_date[0], start_date[1], start_date[2], start_date[3]) + timedelta(hours=1)
    start_date_ = datetime(start_date[0], start_date[1], start_date[2], start_date[3], start_date[4])
    end_date_ = datetime(end_date[0], end_date[1], end_date[2], end_date[3], end_date[4])
    
    label_dict_merged = {}
    while current_date_ <= end_date_:
        date = int(current_date_.strftime('%Y%m%d'))
        if date not in label_dict_merged:
            label_dict_merged[date] = 0
        label_dict_merged[date] = max(label_dict_merged[date], label_dict[int(current_date_.strftime('%Y%m%d%H%M'))])
        current_date_ += timedelta(hours=1)

    return label_dict_merged

def make_label_file(data_path = './data/flare_data.csv', save_path = './data/flare_label_dict.npy'):
    label_dict = transfer_original_into_dict(data_path)
    label_dict = merge_hourly_labels(label_dict)
    np.save(save_path, label_dict)



if __name__ == '__main__':
    make_label_file()