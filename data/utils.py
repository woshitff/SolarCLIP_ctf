import os
import pickle
from tqdm import tqdm
from datetime import datetime, timedelta

import torch
import numpy as np
from astropy.io import fits


def read_pt_image(path):
    return torch.load(path, weights_only=True)


def read_fits_image(path):
    return fits.open(path)[1].data


def save_list(dir_list, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dir_list, file)


def load_list(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    

def transfer_id_to_date(date_id,y_start=2010, m_start = 5, d_start = 1, y_end=2024, m_end = 7, d_end = 1):
    start_date = datetime(y_start, m_start, d_start)
    end_date = datetime(y_end, m_end, d_end)
    if date_id < 0 or date_id > (end_date - start_date).days*24*60:
        raise ValueError(f'date_id out of range, should be in 0 to {(end_date - start_date).days*24*60}')
    # current_date = start_date + timedelta(days=date_id//(24*60))
    current_date = start_date + timedelta(minutes=date_id)

    return current_date

def transfer_date_to_id(y_now, m_now, d_now, y_start=2010, m_start = 5, d_start = 1, y_end=2024, m_end = 7, d_end = 1):
    start_date = datetime(y_start, m_start, d_start)
    end_date = datetime(y_end, m_end, d_end)
    current_date = datetime(y_now, m_now, d_now)

    if current_date < start_date or current_date > end_date:
        raise ValueError(f'current_date out of range, should be in {start_date} to {end_date}')
    date_id = (current_date - start_date).days
    date_id *= 24*60
    return date_id


def get_modal_dir(modal, date_id, y_start=2010, m_start=5, d_start=1, y_end=2024, m_end=7, d_end=1):
    current_date = transfer_id_to_date(date_id, y_start, m_start, d_start, y_end, m_end, d_end)
    date_str_1 = current_date.strftime('%Y/%m/%d')
    date_str_2 = current_date.strftime('%Y%m%d')
    formatted_hours = f"{current_date.hour:02d}"
    formatted_minutes = f"{current_date.minute:02d}"
    if modal == 'magnet':
        path_pt = f"/mnt/nas/home/huxing/202407/nas/data/hmi/magnet_pt/{date_str_1}/hmi.M_720s.{date_str_2}_{formatted_hours}{formatted_minutes}00_TAI.pt"
        path_fits = f"/mnt/nas/home/huxing/202407/nas/data/hmi/fits/hmi.M_720s.{date_str_2}_{formatted_hours}{formatted_minutes}00_TAI.fits"
    elif modal == '0094':
        path_pt = f"/mnt/nas/home/huxing/202407/nas/data/spectral/0094_pt/{date_str_1}/AIA{date_str_2}_{formatted_hours}{formatted_minutes}_0094.pt"
        path_fits = f"/mnt/nas/home/zhouyuqing/downloads/AIA{date_str_2}_{formatted_hours}{formatted_minutes}_0094.fits"
    return path_fits, path_pt


def make_dir_list(modal):
    if modal == 'magnet':
        current_date = datetime(2010, 5, 1)
        one_day = timedelta(days=1)
        end_date = datetime(2024, 6, 30)
        dir_list_fits = []
        dir_list_pt = []
        while current_date <= end_date:
            date_str_1 = current_date.strftime('%Y/%m/%d')
            date_str_2 = current_date.strftime('%Y%m%d')
            for current_date_hours in range(24):
                for current_data_minutes in range(60):
                    formatted_hours = f"{current_date_hours:02d}"
                    formatted_minutes = f"{current_data_minutes:02d}"
                    path_pt = f"/mnt/nas/home/huxing/202407/nas/data/hmi/magnet_pt/{date_str_1}/hmi.M_720s.{date_str_2}_{formatted_hours}{formatted_minutes}00_TAI.pt"
                    path_fits = f"/mnt/nas/home/huxing/202407/nas/data/hmi/fits/hmi.M_720s.{date_str_2}_{formatted_hours}{formatted_minutes}00_TAI.fits"
                    dir_list_pt.append(path_pt)
                    dir_list_fits.append(path_fits)
            current_date += one_day
        if not os.path.exists('./Data/dir_list'):
            os.makedirs('./Data/dir_list')
        save_list(dir_list_pt, './Data/dir_list/magnet_dir_list_pt.pkl')
        save_list(dir_list_fits, './Data/dir_list/magnet_dir_list_fits.pkl')
        print('save_list done')
    elif modal == '0094':
        current_date = datetime(2010, 5, 1)
        one_day = timedelta(days=1)
        end_date = datetime(2024, 6, 30)
        dir_list_fits = []
        dir_list_pt = []
        while current_date <= end_date:
            date_str_1 = current_date.strftime('%Y/%m/%d')
            date_str_2 = current_date.strftime('%Y%m%d')
            for current_date_hours in range(24):
                for current_data_minutes in range(60):
                    formatted_hours = f"{current_date_hours:02d}"
                    formatted_minutes = f"{current_data_minutes:02d}"
                    path_pt = f"/mnt/nas/home/huxing/202407/nas/data/spectral/0094_pt/{date_str_1}/AIA{date_str_2}_{formatted_hours}{formatted_minutes}_0094.pt"
                    path_fits = f"/mnt/nas/home/zhouyuqing/downloads/AIA{date_str_2}_{formatted_hours}{formatted_minutes}_0094.fits"
                    # /mnt/nas/home/zhouyuqing/downloads/AIA20100501_0000_0094.fits
                    dir_list_pt.append(path_pt)
                    dir_list_fits.append(path_fits)
            current_date += one_day
        if not os.path.exists('./Data/dir_list'):
            os.makedirs('./Data/dir_list')
        save_list(dir_list_pt, './Data/dir_list/0094_dir_list_pt.pkl')
        save_list(dir_list_fits, './Data/dir_list/0094_dir_list_fits.pkl')
        print('save_list done')


def transfer_fits_to_pt(modal, exist_list=None, time_interval = [0,7452000]):

    if not os.path.exists('./Data/idx_list'):
        os.makedirs('./Data/idx_list')

    if exist_list is None:
        exist_list = np.zeros(time_interval[1], dtype=np.bool)
    else:
        exist_list = load_list(exist_list)
        print(len(exist_list))
    
    move_num = 0
    for i in tqdm(range(time_interval[0], time_interval[1])):

        if exist_list[i] == 0:  # no pt file now
            dir_fits, dir_pt = get_modal_dir(modal, i)

            if os.path.exists(dir_fits):
                try:
                    fits_img = read_fits_image(dir_fits)
                    fits_img = np.nan_to_num(fits_img, nan=0.0)
                    pt_img = torch.tensor(fits_img,dtype=torch.float32)
                    pt_dir = os.path.dirname(dir_pt)
                    if not os.path.exists(pt_dir):
                        os.makedirs(pt_dir)
                    torch.save(pt_img, dir_pt)
                    exist_list[i] = True
                    move_num += 1
                except:
                    pass
        if i % ((time_interval[1]-time_interval[0])//100) == 0:
            save_list(exist_list, f'./Data/idx_list/{modal}_exist_idx.pkl')

    print(f'transfer done, {move_num} files transfterd, exist list saved to ./Data/idx_list/{modal}_exist_idx.pkl')

def transfer_float64_to_float32(modal, exist_list=None, time_interval = [0,7452000]):

    if not os.path.exists('./Data/idx_list'):
        os.makedirs('./Data/idx_list')

    if exist_list is None:
        exist_list = np.zeros(time_interval[1], dtype=np.bool)
    else:
        exist_list = load_list(exist_list)
        print(len(exist_list))
    move_num = 0
    for i in tqdm(range(time_interval[0], time_interval[1])):

        if exist_list[i] == 1:  # no float32 file now
            dir_fits, dir_pt = get_modal_dir(modal, i)

            if os.path.exists(dir_fits):
                try:
                    float64_img = torch.load(dir_pt)
                    float32_img = float64_img.to(torch.float32)
                    pt_dir = os.path.dirname(dir_pt)
                    if not os.path.exists(pt_dir):
                        os.makedirs(pt_dir)
                    torch.save(float32_img, dir_pt)
                    exist_list[i] = True
                    move_num += 1
                except:
                    pass
        if i % ((time_interval[1]-time_interval[0])//10) == 0:
            save_list(exist_list, f'./Data/idx_list/{modal}_exist_idx.pkl')

    print(f'transfer done, {move_num} files transfterd to float32, exist list saved to ./Data/idx_list/{modal}_exist_idx.pkl')

def clean_error_dir(fits_dir_list):
    for i in tqdm(range(len(fits_dir_list))):
        fits_dir = fits_dir_list[i]
        pt_dir = fits_dir.replace('fits', 'pt')
        if os.path.exists(pt_dir):
            os.rmdir(pt_dir)
    print('clean done')


def update_exist_list(modal, save_dir = './Data/idx_list',time_interval = [0,7452000]):
    exist_idx = np.zeros(time_interval[1], dtype=np.bool)
    if modal == 'magnet':
        for i in tqdm(range(time_interval[0], time_interval[1])):
            dir_fits, dir_pt = get_modal_dir(modal, i)
            if os.path.exists(dir_pt):
                exist_idx[i] = True
        save_list(exist_idx, f'{save_dir}/{modal}_exist_idx.pkl')
    elif modal == '0094':
        for i in tqdm(range(time_interval[0], time_interval[1])):
            dir_fits, dir_pt = get_modal_dir(modal, i)
            if os.path.exists(dir_pt):
                exist_idx[i] = True
        save_list(exist_idx, f'{save_dir}/{modal}_exist_idx.pkl')



if __name__ == '__main__':

    # make_dir_list('magnet')
    # dir_list_fits = load_list('./Data/dir_list/magnet_dir_list_fits.pkl')
    # dir_list_pt = load_list('./Data/dir_list/magnet_dir_list_pt.pkl')
    # magnet_exist_idx = load_list('./Data/idx_list/magnet_exist_idx.pkl')
    # with open('/mnt/nas/home/huxing/202407/ctf/SolarCLIP/Data/idx_list/magnet_exist_idx.pkl', 'rb') as f:
    #     magnet_exist_idx = pickle.load(f)
    
    # transfer_fits_to_pt(modal='magnet',exist_list=magnet_exist_idx)

    # # make_dir_list('0094')
    # dir_list_fits = load_list('./Data/dir_list/0094_dir_list_fits.pkl')
    # dir_list_pt = load_list('./Data/dir_list/0094_dir_list_pt.pkl')
    # # exist_idx = load_list('./Data/idx_list/0094_exist_idx.pkl')
    # exist_idx = None
    # transfer_fits_to_pt('0094', dir_list_fits, dir_list_pt, exist_idx)

    # update_exist_list('magnet')

    # pass
    

    # transfer_fits_to_pt('0094',exist_list='./Data/idx_list/0094_exist_idx.pkl')
    transfer_fits_to_pt('magnet',exist_list='./Data/idx_list/magnet_exist_idx.pkl')