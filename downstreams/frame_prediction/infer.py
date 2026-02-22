import os
import torch
from omegaconf import OmegaConf
from datetime import datetime, timedelta

import torch.signal

from models.video_models.solar_modeling import SolarVivit
from downstreams.frame_prediction.utils import get_image_from_time, get_id_from_time, get_lalo_from_time, plot_solar_image, plot_solar_images_list

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    if "A40" in gpu_name or "A100" in gpu_name:
        torch.set_float32_matmul_precision('high') # highest, high, medium
        print(f'device is {gpu_name}, set float32_matmul_precision to high')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


base_config = OmegaConf.create({
    'solarmodel': {
        'type': 'solarvivit',
        'load_dir': '/mnt/tianwen-tianqing-nas/tianwen/home/gaozhe/codes/solar-transformer/logs/base/notrainer_1-64-64_6p1-16layer/best_repeat',
        'config': {}
    }
})

# Load the model as a global variable
model = SolarVivit.from_pretrained(base_config.solarmodel.load_dir)
model.eval()
model.to(device)


def time_to_date(time: int):
    year = time // 10000000000
    month = (time // 100000000) % 100
    day = (time // 1000000) % 100
    hour = (time // 10000) % 100
    minute = (time // 100) % 100

    date = datetime(year, month, day, hour, minute) # type: datetime
    return date

def date_to_time(date: datetime):
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute

    time = year * 10000000000 + month * 100000000 + day * 1000000 + hour * 10000 + minute * 100
    return time

def make_input(time_list: list, predict_len: int = 1):
    # time_list: [6]
    # video_in [1, 6, 1, 1024, 1024], date_ids [1, 6+predlen], LaLos [1, 6+predlen, 2]
    input_len = 6
    time_step = timedelta(hours=1)
    assert len(time_list) == input_len
    time_list_ = [time_to_date(time) for time in time_list]
    for i in range(input_len-1):
        assert time_list_[i] + time_step == time_list_[i+1], f"error in time list: {time_list_}"

    for i in range(predict_len):
        time_list_.append(time_list_[-1] + time_step)
    time_list_ = [date_to_time(time) for time in time_list_]
    
    video_in = torch.stack([get_image_from_time(time) for time in time_list])
    date_ids = torch.tensor([get_id_from_time(time) for time in time_list_])
    LaLos = torch.stack([get_lalo_from_time(time) for time in time_list_])
    
    video_in = video_in.unsqueeze(0) # [1, 6, 1024, 1024]
    video_in = video_in.unsqueeze(2) # [1, 6, 1, 1024, 1024]
    date_ids = date_ids.unsqueeze(0) # [1, 7]
    LaLos = LaLos.unsqueeze(0) # [1, 7, 2]
    video_in = video_in.to(device)
    date_ids = date_ids.to(device)
    LaLos = LaLos.to(device)
    
    return video_in, date_ids, LaLos


def long_time_prediction(time_list: list, predict_len: int = 1, save_dir = None):

    video_in, date_ids, LaLos = make_input(time_list, predict_len)
    with torch.no_grad():
        output = model.continuous_infer(predict_length=predict_len, video_in=video_in, date_ids=date_ids, LaLos=LaLos) # (b=1, predict_len, c=1, H, W)

    # log1p enhanced, deenhance
    output = torch.sign(output) * torch.expm1(torch.abs(output))
    output = output.cpu()
    output = output.squeeze(0) # (predict_len, 1, 1024, 1024)
    output = output.squeeze(1) # (predict_len, 1024, 1024)
    video_in = video_in.cpu()
    video_in = video_in.squeeze(0) # (6, 1, 1024, 1024)
    video_in = video_in.squeeze(1) # (6, 1024, 1024)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(output, os.path.join(save_dir, 'output.pt'))
        torch.save(video_in, os.path.join(save_dir, 'video_in.pt'))
    return output, video_in

if __name__ == '__main__':
    time_list = [20230412060000, 20230412070000, 20230412080000, 20230412090000, 20230412100000, 20230412110000]
    predict_len = 4
    save_dir = '/mnt/tianwen-tianqing-nas/tianwen/home/gaozhe/codes/solar-transformer/downstreams/frame_prediction'
    output, video_in = long_time_prediction(time_list, predict_len)
    images = torch.cat([video_in, output], dim=0) # (10, 1024, 1024)
    plot_solar_images_list(images, log1p=True, save_dir=save_dir, cbar = True, v_tuple = [-8,8])