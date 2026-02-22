import pandas as pd
import numpy as np
import wget
import os
from astropy.io import fits

import torch
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from Data.utils import transfer_date_to_id, get_modal_dir
from datetime import datetime

def read_image(path):
    if path.endswith(".pt"):
        return torch.load(path, weights_only=True)
    elif path.endswith(".fits"):
        return fits.open(path)[1].data
    else:
        raise ValueError("Unsupported file format. Only .pt and .fits are supported.")


def get_url_from_time(time :int,
                      modal: str):
    year = time // 10000000000
    month = (time // 100000000) % 100
    day = (time // 1000000) % 100
    hour = (time // 10000) % 100
    minute = (time // 100) % 100

    if modal == 'hmi':
        url = f'https://jsoc1.stanford.edu/data/hmi/fits/{year:04d}/{month:02d}/{day:02d}/hmi.M_720s.{year:04d}{month:02d}{day:02d}_{hour:02d}0000_TAI.fits'
                # https://jsoc1.stanford.edu/data/hmi/fits/2011/02/02/hmi.M_720s.20110202_000001_TAI.fits
    else:
        url = f'https://jsoc1.stanford.edu/data/aia/synoptic/{year:04d}/{month:02d}/{day:02d}/H{hour:02d}00/AIA{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}_{modal}.fits'
    
    return url

def get_path_from_time(time :int,
                       modal: str):
    year = time // 10000000000
    month = (time // 100000000) % 100
    day = (time // 1000000) % 100
    hour = (time // 10000) % 100
    minute = (time // 100) % 100

    if modal == 'hmi':
        path_fits = f"/mnt/tianwen-tianqing-nas/tianwen/ctf/data/hmi/{modal}_fits/{modal}.M_720s.{year:04d}{month:02d}{day:02d}_{hour:02d}0000_TAI.fits"
        path_pt = f"/mnt/tianwen-tianqing-nas/tianwen/ctf/data/hmi/{modal}_pt/{modal}.M_720s.{year:04d}{month:02d}{day:02d}_{hour:02d}0000_TAI.pt"
    else:
        path_fits = f"/mnt/tianwen-tianqing-nas/tianwen/ctf/data/aia/{modal}_fits/AIA{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}_{modal}.fits"
        path_pt = f"/mnt/tianwen-tianqing-nas/tianwen/ctf/data/aia/{modal}_pt/AIA{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}_{modal}.pt"

    return path_fits, path_pt


def get_image_from_time(time: int = 20250228120000,
                        modal: str = 'hmi'):
    
    path_fits, path_pt = get_path_from_time(time, modal)

    if os.path.isfile(path_pt):
        img = read_image(path_pt)
        return img
    else:
        try:
            url = get_url_from_time(time, modal)
            if not os.access(os.path.dirname(path_fits), os.W_OK):
                print("无写入权限！")
            # wget.download(url, path_fits)
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(path_fits, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"{url}下载失败，状态码：{response.status_code}")
            fits_img = read_image(path_fits)
            fits_img = np.nan_to_num(fits_img, nan=0.0)
            pt_img = torch.tensor(fits_img,dtype=torch.float32)
            pt_dir = os.path.dirname(path_pt)
            if not os.path.exists(pt_dir):
                os.makedirs(pt_dir)
            torch.save(pt_img, path_pt)
            print(f'{path_pt} download sucessfully!')
            return pt_img

        except Exception as e:
            print(f"There is no image for {time} in {modal}")


def get_id_from_time(time: int = 20250228120000):
    year = time // 10000000000
    month = (time // 100000000) % 100
    day = (time // 1000000) % 100
    hour = (time // 10000) % 100
    minute = (time // 100) % 100

    idx = transfer_date_to_id(year, month, day, hour, minute)
    return idx # shape: 1,

def get_lalo_from_time(time: int = 20250228120000):
    year = time // 10000000000
    month = (time // 100000000) % 100
    day = (time // 1000000) % 100
    hour = (time // 10000) % 100
    minute = (time // 100) % 100

    date = datetime(year, month, day, hour, minute) # type: datetime
    date_query_str = [date.strftime("%Y.%m.%d_%H:%M:%S_TAI")]
    
    # step 1, read the lalo csv
    lalo_df = pd.read_csv("Data/LatLonXsl/2010-2024-1h-filled-deduped.csv")

    # step 2, get lalo
    lal0 = lalo_df[lalo_df['T_REC'].isin(date_query_str)][['CRLN_OBS', 'CRLT_OBS']].to_numpy()

    lal0 = torch.from_numpy(lal0).squeeze(0).float()
    
    return lal0 # shape: 2,

# draw the sun
def plot_solar_image(data, log1p = True, title = "", v_tuple = None, mask=None, save_dir = None):
    # data: (1024, 1024)
    if type(data) == torch.Tensor:
        data = data.numpy()
    if log1p:
        data = np.log1p(data)
    if v_tuple is None:
        v_max = np.max(np.abs(data))
        v_min = -v_max
        v_tuple = (v_min, v_max)
    if mask is not None:
        data = data+mask * v_tuple[0]
    plt.imshow(data, cmap='RdBu', vmin=v_tuple[0], vmax= v_tuple[1],interpolation='none')
    cbar = plt.colorbar()
    cbar.set_label('Pixel Value') 
    plt.title(title)
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.show()
    plt.close()

def plot_solar_images_list(data_list, log1p = True, title = "", cbar = False, v_tuple = None, mask=None, save_dir = None):
    # data_list: [n, 1024, 1024]
    if type(data_list) == torch.Tensor:
        data_list = data_list.numpy()
    if log1p:
        data_list = np.sign(data_list) * np.log1p(np.abs(data_list))
    if v_tuple is None:
        v_max = np.max(np.abs(data_list))
        v_min = -v_max
        v_tuple = (v_min, v_max)

    # 创建子图
    n = len(data_list)
    fig, axs = plt.subplots(1, n, figsize=(n*3, 3))  # 调整 figsize 以适应多个子图
    if n == 1:
        axs = [axs]
        
    for i, data in enumerate(data_list):
        # 绘制图像
        if mask is not None:
            data = data+mask * v_tuple[0]
        im = axs[i].imshow(data, cmap='RdBu', vmin=v_tuple[0], vmax=v_tuple[1], interpolation='none')
        if cbar:
            cbar = plt.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
            cbar.set_label('Pixel Value')
        axs[i].axis('off')
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    # 调整子图布局
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'output.png'))
    plt.show()
    plt.close()


if __name__ == '__main__':
    pass
