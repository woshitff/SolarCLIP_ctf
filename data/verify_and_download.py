import pickle
import wget
import os 
import time

import numpy as np
import torch

from tqdm import tqdm

from utils import transfer_id_to_date, get_modal_dir, read_fits_image, transfer_id_to_date_V2, get_modal_dir_V2

import concurrent.futures

def verify_and_download(modal, exist_idx_list, time_interval = [0,1e+32]):
    download_num = 0
    error_url = []
    
    for i in tqdm(range(len(exist_idx_list))):

        if i<time_interval[0] or i>time_interval[1]:
            continue

        if exist_idx_list[i] == 0: # no pt file now
            date_time = transfer_id_to_date(i)
            year = date_time.year
            month = date_time.month
            day = date_time.day
            hour = date_time.hour
            minute = date_time.minute

            if modal == 'magnet':
                if minute == 0:
                    url = f'http://jsoc1.stanford.edu/data/hmi/fits/{year:04d}/{month:02d}/{day:02d}/hmi.M_720s.{year:04d}{month:02d}{day:02d}_{hour:02d}0000_TAI.fits'
                    # http://jsoc.stanford.edu/data/hmi/fits/2011/02/02/hmi.M_720s.20110202_000001_TAI.fits
                    dir_fits = get_modal_dir('magnet',i)[0]
                    try:
                        wget.download(url, dir_fits) # download fits file
                        download_num += 1
                    except Exception as e:
                        error_url.append(url)
            else:
                raise ValueError('modal not supported')

    print(f'{download_num} files downloaded, {len(error_url)} files failed:')
    for url in error_url:
        print(url)


def dl_and_convert(modal, exist_idx_list, time_interval = [0,1e+32]):
    download_num = 0
    error_url = []
    
    for i in tqdm(range(len(exist_idx_list))):

        if i<time_interval[0] or i>time_interval[1]:
            continue

        if exist_idx_list[i] == 0: # no pt file now
            date_time = transfer_id_to_date(i)
            year = date_time.year
            month = date_time.month
            day = date_time.day
            hour = date_time.hour
            minute = date_time.minute

            if modal == 'magnet':
                if minute == 0:
                    url = f'http://jsoc1.stanford.edu/data/hmi/fits/{year:04d}/{month:02d}/{day:02d}/hmi.M_720s.{year:04d}{month:02d}{day:02d}_{hour:02d}0000_TAI.fits'
                    # http://jsoc1.stanford.edu/data/hmi/fits/2011/02/02/hmi.M_720s.20110202_000001_TAI.fits
                    dir_fits, dir_pt = get_modal_dir('magnet',i)
                    try:
                        wget.download(url, dir_fits) # download fits file
                        download_num += 1
                        try:
                            fits_img = read_fits_image(dir_fits)
                            fits_img = np.nan_to_num(fits_img, nan=0.0)
                            pt_img = torch.tensor(fits_img,dtype=torch.float32)
                            pt_dir = os.path.dirname(dir_pt)
                            if not os.path.exists(pt_dir):
                                os.makedirs(pt_dir)
                            torch.save(pt_img, dir_pt)
                            # exist_idx_list[i] = True
                        except Exception as e:
                            print(f"Error occured : {e}, delete {dir_pt} if exists")
                            if os.path.exists(dir_pt):
                                os.remove(dir_pt)                           
                    except Exception as e:
                        error_url.append(url)
            else:
                raise ValueError('modal not supported')

    print(f'{download_num} files downloaded, {len(error_url)} files failed:')
    for url in error_url:
        print(url)
            

# 2025/02/06 version 2： one picture for each aia modal per day
def dl_and_conver_V2(modal, 
                    #  pt_idx_list, 
                     time_interval = [0, 1e+32]):
    """
    modal: hmi, 0094, 0131, 0171, 0193, 0211, 0304, 0335, 1600, 1700, 4500
    """
    print('start')
    exist_num = 0
    download_num = 0
    error_url = []
    error_path = '/mnt/tianwen-tianqing-nas/tianwen/ctf/data/aia/'
    if not os.path.exists(error_path):
        os.mkdir(error_path)

    pbar = tqdm(range(6000))
    for i in pbar:
        if i<time_interval[0] or i>time_interval[1]:
            continue
        
        # if pt_idx_list[i] == 0: # no pt file now
        date_time = transfer_id_to_date_V2(i)
        year = date_time.year
        month = date_time.month
        day = date_time.day
        pbar.set_description(f'{modal} | {year} | {month} | {day}')


        if not modal == 'hmi':
            url = f'https://jsoc1.stanford.edu/data/aia/synoptic/{year:04d}/{month:02d}/{day:02d}/H0000/AIA{year:04d}{month:02d}{day:02d}_0000_{modal}.fits'
                    # https://jsoc1.stanford.edu/data/hmi/fits/2011/02/02/hmi.M_720s.20110202_000001_TAI.fits
        else:
            url = f'https://jsoc1.stanford.edu/data/hmi/fits/{year:04d}/{month:02d}/{day:02d}/hmi.M_720s.{year:04d}{month:02d}{day:02d}_000000_TAI.fits'
        path_fits, path_pt = get_modal_dir_V2(modal, i)

        try:
            dir_fits = os.path.dirname(path_fits)
            if not os.path.exists(dir_fits):
                os.makedirs(dir_fits)
            if os.path.exists(path_fits):
                exist_num += 1
            else:
                wget.download(url, path_fits) # download fits file
                download_num += 1
            try:
                if not os.path.exists(path_pt):
                    fits_img = read_fits_image(path_fits)
                    fits_img = np.nan_to_num(fits_img, nan=0.0)
                    pt_img = torch.tensor(fits_img,dtype=torch.float32)

                    pt_dir = os.path.dirname(path_pt)
                    if not os.path.exists(pt_dir):
                        os.makedirs(pt_dir)
                    torch.save(pt_img, path_pt)
                    # pt_idx_list[i] = True
            except Exception as e:
                print(f"Error occured : {e}, delete {path_pt} if exists")
                if os.path.exists(path_pt):
                    os.remove(path_pt)                           
        except Exception as e:
            error_url.append(url)
            with open(f'{error_path}error_url.txt', 'a') as f:
                f.writelines(f'{error_url[-1]}\n')
    
        print(f'|  {download_num} success, {len(error_url)} fail, {modal} {year} {month} {day}')
        # time.sleep(5)

    for url in error_url:
        print(url)
    print('finish')
            
# 2025/02/07 get the exist pickle


if __name__ == '__main__' :

    # with open('/mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/Data/idx_list/magnet_exist_idx.pkl','rb') as f:
    #     exist_idx_list = pickle.load(f)
    # i = 13
    # print(500000*i,500000*(i+1))
    # print('start date :', transfer_id_to_date(500000*i))
    # print('end date :', transfer_id_to_date(500000*(i+1)))

    # dl_and_convert('magnet',exist_idx_list,[500000*i,500000*(i+1)])
    # with open(f'/mnt/tianwen-tianqing-nas/tianwen/ctf/solarclip/ctf_105/SolarCLIP_ctf/data/idx_list_v2/{modal}_exist_idx.pkl','rb') as f:
    #     pt_idx_list = pickle.load(f)
    # modal_list = ['0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '1700', '4500']
    # with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
    #     executor.map(dl_and_conver_V2, modal_list)

    dl_and_conver_V2('hmi')