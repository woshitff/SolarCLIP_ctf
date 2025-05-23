import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib

import torch
from omegaconf import OmegaConf
import argparse

from data.download_api import get_image_from_time
from models.clipmodels.solarclip import SolarCLIP


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

def parse_args():
    parser = argparse.ArgumentParser(description="Modal Transfer Script")
    
    parser.add_argument('--config_path', type=str, default='configs/train_configs/reconmodels/ldm/mainconfig/bimodal_trans/0094_other/new_0094to0131.yaml'
,
                        help='Path to the YAML config file.')
    parser.add_argument('--time', type=int, default=202504011200,
                        help='Timestamp to process. Format: YYYYMMDDHHMM')
    parser.add_argument('--input_modal', type=str, default='0094',
                        help='Name of the input modality.')
    parser.add_argument('--output_modal', type=str, default='0131',
                        help='Name of the output modality.')
    parser.add_argument('--save_dir', type=str, default='./infer_results',
                        help='Directory to save output.')

    return parser.parse_args()

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    if "A40" in gpu_name or "A100" in gpu_name:
        torch.set_float32_matmul_precision('high') # highest, high, medium
        print(f'device is {gpu_name}, set float32_matmul_precision to high')


model = None

def load_model(config: OmegaConf):
    global model

    model = instantiate_from_config(config.model)
    model.eval()
    print("Model loaded successfully.")


def load_data(time: int, input_modal: str):
    input_data = get_image_from_time(time, input_modal)
    input_data = input_data.unsqueeze(0).unsqueeze(0)
    
    return input_data

def modal_transfer(time: int, input_modal: str, output_modal: str, save_dir: str = None):
    input_data = load_data(time, input_modal)

    global model
    if model is None:
        raise ValueError("Model is not loaded. Please call load_model first.")
    output_data = model.infer(input_data)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{time}_{output_modal}.pt"
        save_path = os.path.join(save_dir, filename)
        torch.save(output_data, save_path)
        print(f"[INFO] Output saved to {save_path}")
    
    return output_data, input_data

if __name__ == '__main__':
    args = parse_args()
    config = OmegaConf.load(args.config_path)  

    load_model(config)  
    
    output, input = modal_transfer(args.time, input_modal=args.input_modal, output_modal=args.output_modal, save_dir=args.save_dir)

    from train_scripts.visualization.solarplot import solarplot, format_timestamp
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        output_name = f"{args.time}_{args.output_modal}.png"
        output_path = os.path.join(args.save_dir, output_name)
        print(f"[INFO] Output saved to {output_path}")

        input_name = f"{args.time}_{args.input_modal}.png"
        input_path = os.path.join(args.save_dir, input_name)
        print(f"[INFO] Output saved to {input_path}")
        
    solarplot(output.squeeze(0).squeeze(0).cpu().numpy(), args.output_modal, format_timestamp(args.time), output_path)
    solarplot(input.squeeze(0).squeeze(0).cpu().numpy(), args.input_modal, format_timestamp(args.time), input_path)
