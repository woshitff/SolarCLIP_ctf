import pickle
import wget
import os 
import time

import numpy as np
import torch
from astropy.io import fits

def read_image(path):
    if path.endswith(".pt"):
        return torch.load(path, weights_only=True)
    elif path.endswith(".fits"):
        return fits.open(path)[1].data
    else:
        raise ValueError("Unsupported file format. Only .pt and .fits are supported.")


def get_url_from_time(time :int,
                      modal: str):
    year = time // 100000000
    month = (time // 1000000) % 100
    day = (time // 10000) % 100
    hour = (time // 100) % 100
    minute = (time // 1) % 100

    if modal == 'hmi':
        url = f'https://jsoc1.stanford.edu/data/hmi/fits/{year:04d}/{month:02d}/{day:02d}/hmi.M_720s.{year:04d}{month:02d}{day:02d}_{hour:02d}0000_TAI.fits'
                # https://jsoc1.stanford.edu/data/hmi/fits/2011/02/02/hmi.M_720s.20110202_000001_TAI.fits
    else:
        url = f'https://jsoc1.stanford.edu/data/aia/synoptic/{year:04d}/{month:02d}/{day:02d}/H{hour:02d}00/AIA{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}_{modal}.fits'
    
    return url

def get_path_from_time(time :int, 
                       modal: str):
    year = time // 100000000
    month = (time // 1000000) % 100
    day = (time // 10000) % 100
    hour = (time // 100) % 100
    minute = (time // 1) % 100

    if modal == 'hmi':
        path_fits = f"/mnt/tianwen-tianqing-nas/tianwen/ctf/data/hmi/{modal}_fits/{modal}.M_720s.{year:04d}{month:02d}{day:02d}_{hour:02d}0000_TAI.fits"
        path_pt = f"/mnt/tianwen-tianqing-nas/tianwen/ctf/data/hmi/{modal}_pt/{modal}.M_720s.{year:04d}{month:02d}{day:02d}_{hour:02d}0000_TAI.pt"
    else:
        path_fits = f"/mnt/tianwen-tianqing-nas/tianwen/ctf/data/aia/{modal}_fits/AIA{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}_{modal}.fits"
        path_pt = f"/mnt/tianwen-tianqing-nas/tianwen/ctf/data/aia/{modal}_pt/AIA{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}_{modal}.pt"

    return path_fits, path_pt


def get_image_from_time(time: int = 202502281200,
                        modal: str = '0094'):
    
    path_fits, path_pt = get_path_from_time(time, modal)

    if os.path.isfile(path_pt) or os.path.isfile(path_fits):
        img = read_image(path_pt)
        return img
    else:
        try:
            url = get_url_from_time(time, modal)
            wget.download(url, path_fits)
            fits_img = read_image(path_fits)
            fits_img = np.nan_to_num(fits_img, nan=0.0)
            pt_img = torch.tensor(fits_img,dtype=torch.float32)
            pt_dir = os.path.dirname(path_pt)
            if not os.path.exists(pt_dir):
                os.makedirs(pt_dir)
            torch.save(pt_img, path_pt)
            return pt_img

        except Exception as e:
            print(f"There is no image for {time} in {modal}")

    # img = torch.rand(1, 1024, 1024)
    # return img # shape: 1, 1024, 1024

if __name__ == '__main__':
    t = 202502281200
    img = get_image_from_time(time=t, modal='0094')