import os

import torch
from datetime import datetime, timedelta


from omegaconf import OmegaConf

from models.video_models.solar_modeling import SolarVivit
from .model import Predictor, SolarPredictor
from .dataset import ClassificationDataset as SolarDataset

from .utils import get_image_from_time, get_id_from_time, get_lalo_from_time

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    if "A40" in gpu_name or "A100" in gpu_name:
        torch.set_float32_matmul_precision('high') # highest, high, medium
        print(f'device is {gpu_name}, set float32_matmul_precision to high')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


base_config = OmegaConf.create({
    'training': {
        'load_dir': None,
        'epochs': 1000,
        'log_dir': '/mnt/tianwen-tianqing-nas/tianwen/home/gaozhe/codes/solar-transformer/downstreams/flare_prediction/logs/logs',
        'save_dir': '/mnt/tianwen-tianqing-nas/tianwen/home/gaozhe/codes/solar-transformer/downstreams/flare_prediction/logs/output',
        'early_stop': 0,
        'enable_checkpointing': True,
        'precision': 'bf16',
        'strategy': 'ddp_find_unused_parameters_true',
        'checkpoint': {
            "save_top_k": 1,
            'every_n_train_steps': None,
            "every_n_epochs": 1,
        }
    },
    'solarmodel': {
        'type': 'solarvivit',
        'load_dir': '/mnt/tianwen-tianqing-nas/tianwen/home/gaozhe/codes/solar-transformer/logs/base/notrainer_1-64-64_6p1-16layer/best_repeat',
        'config': {}
    },
    'predictor': {
        'load_dir': None,
        'num_classes': 6,
        'hidden_size': 768,
        'hidden_layer': 6
    },
    'solarpredictor': {
        'only_train_predictor': False,
        'criterion': 'CrossEntropyLoss',
        'optimizer': 'adamw',
        'lr': 1e-6,
        'scheduler': {
            'type': 'cosine',
            'params': {
                'T_max': 1000,
                'eta_min': 0,
            },
            'interval': 'epoch'
        }
    },
    'data_configs': {
        'train_interval': [[2010,5,1,1,34], [2015,6,27,21,19]],
        'eval_interval': [[2015,6,27,21,20], [2016,6,27,21,19]],
        "time_step": 1,
        "num_frames": 7,
        "load_imgs": False,
        "label_type": "ABCMX",
        "label_dir": "/mnt/tianwen-tianqing-nas/tianwen/home/gaozhe/codes/solar-transformer/downstreams/flare_prediction/data/flare_label_dict.csv",
        'batch_size': 32,
        'num_workers': 0,
        'pin_memory': False,
    },
    "logger": {
        "type": "wandb",
        "config": {
            "name": "full_alltoken",
            "project": "flare_prediction",
            "key": "99f8f0875da5df4e119430dcdd439c09847cb990",
            "entity": "nankai_remote",
        }
    }
})

# add model in the global scope

model_dir: str = '/mnt/tianwen-tianqing-nas/tianwen/home/gaozhe/codes/solar-transformer/downstreams/flare_prediction/logs/output/best.ckpt'
# config_dir: str = '/mnt/tianwen-tianqing-nas/tianwen/home/gaozhe/codes/solar-transformer/downstreams/flare_prediction/logs/logs/config.yaml'
config_dir = None

try:
    config = OmegaConf.load(config_dir)
    config = OmegaConf.merge(base_config, config)
except:
    print('Invalid config file, using default config')
    config = base_config

# load model
if config.solarmodel.type == 'SolarVivit' or config.solarmodel.type == 'solarvivit':
    if config.solarmodel.load_dir:
        try:
            model = SolarVivit.from_pretrained(config.solarmodel.load_dir)
            config.solarmodel.config = OmegaConf.merge(config.solarmodel.config, model.config.to_dict())
        except:
            print('Failed to load model from checkpoint, using default model')
            model = SolarVivit(**config.solarmodel.config)
    else:
        model = SolarVivit(**config.solarmodel.config)
else:
    raise ValueError('Invalid model type')
predictor = Predictor(**config.predictor)
if config.predictor.load_dir:
    predictor.load_state_dict(torch.load(config.predictor.load_dir))
solar_predictor = SolarPredictor.load_from_checkpoint(model_dir, model=model, predictor=predictor)
solar_predictor.eval()
solar_predictor.freeze()
solar_predictor.to(device)


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

def make_input(time_list: list):
    # time_list: [6]
    # video_in [1, 6, 1, 1024, 1024], date_ids [1, 6+1], LaLos [1, 6+1, 2]
    input_len = 6
    time_step = timedelta(hours=1)
    assert len(time_list) == input_len
    time_list_ = [time_to_date(time) for time in time_list]
    for i in range(input_len-1):
        assert time_list_[i] + time_step == time_list_[i+1], f"error in time list: {time_list_}"
    
    video_in = torch.stack([get_image_from_time(time) for time in time_list])
    # add predict id and lalo
    date_ids = torch.tensor([get_id_from_time(time) for time in time_list]+[get_id_from_time(date_to_time(time_list_[-1] + time_step))])
    LaLos = torch.stack([get_lalo_from_time(time) for time in time_list]+[get_lalo_from_time(date_to_time(time_list_[-1] + time_step))])

    video_in = video_in.unsqueeze(0) # [1, 6, 1024, 1024]
    video_in = video_in.unsqueeze(2) # [1, 6, 1, 1024, 1024]
    date_ids = date_ids.unsqueeze(0) # [1, 7]
    LaLos = LaLos.unsqueeze(0) # [1, 7, 2]
    video_in = video_in.to(device)
    date_ids = date_ids.to(device)
    LaLos = LaLos.to(device)
    
    return video_in, date_ids, LaLos

def flare_prediction(time_list: list):
    video_in, date_ids, LaLos = make_input(time_list)
    with torch.no_grad():
        logit = solar_predictor(video_in, date_ids, LaLos)
        logit = logit[0].squeeze(0) # [6]
        prob = torch.softmax(logit, dim = -1).detach().cpu().numpy()
        prediction = torch.argmax(logit).detach().cpu().numpy()
    
    return prediction, prob


if __name__ == '__main__':
    time_list = [20230412060000, 20230412070000, 20230412080000, 20230412090000, 20230412100000, 20230412110000]
    prediction, prob = flare_prediction(time_list)
    print(prediction)
    print(prob)
