import os

import torch
from omegaconf import OmegaConf
import argparse

from data.download_api import get_image_from_time
from models.clipmodels.solarclip import SolarCLIP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--time', type=int, required=True, help='Timestamp for modal transfer')
    parser.add_argument('--input_modal', type=str, required=True, help='Input modal type')
    parser.add_argument('--output_modal', type=str, required=True, help='Output modal type')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save output')
    return parser.parse_args()

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    if "A40" in gpu_name or "A100" in gpu_name:
        torch.set_float32_matmul_precision('high') # highest, high, medium
        print(f'device is {gpu_name}, set float32_matmul_precision to high')


model = None

def load_model(config: OmegaConf):
    global model
    model_config = config.get('model', {})
    
    model = SolarCLIP.from_pretrained(model_config.get('load_dir', ''))
    model.eval()
    print("Model loaded successfully.")

def modal_transfer(time: int, input_modal: str, output_modal: str, save_dir: str = None):
    global model
    if model is None:
        raise ValueError("Model is not loaded. Please call load_model first.")
    
    input_data = get_image_from_time(time, input_modal)
    output_data = model.infer(input_data)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(output_data, os.path.join(save_dir, 'output.pt'))
    
    return output_data

if __name__ == '__main__':
    args = parse_args()
    config = OmegaConf.load(args.config_path)  

    load_model(config)  
    
    time = 202502281200
    output = modal_transfer(args.time, input_modal=args.input_modal, output_modal=args.output_modal, save_dir=args.save_dir)
    print(output.shape)

    from train_scripts.visualization import solarplot
    solarplot(output, args.output_modal, time, args.save_dir)
