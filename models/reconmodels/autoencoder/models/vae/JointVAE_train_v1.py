# 2025/02/08 Suppose all 11 modals can be loaded in a single machine.
import math
import os
import random
import yaml
from omegaconf import OmegaConf
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.reconmodels.autoencoder.util import config_optimizers
from models.reconmodels.autoencoder.util import instantiate_from_config

# config
# 1. model config
# 2. data config
# 3. train config

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)    
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="*",
        metavar="",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=['configs/train_configs/reconmodels/autoencoder/vae/JointVAE.yaml']
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs/reconmodels/autoencoder/vae",
        help="log directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    return parser


if __name__ == "__main__":
    # ckpt_paths should have same order with data make config yaml can be load from cli.
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    config = OmegaConf.load(opt.config) 


    #### init data
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    train_dataloader = data.train_dataloader
    val_dataloader = data.val_dataloader


    #### Init basic traing config
    training_config = config.training
    device = training_config.device
    epochs = training_config.epochs
    test_epoch = epochs//training_config.test_freq
    save_epoch = epochs//training_config.save_freq


    #### Init Model
    Model = instantiate_from_config(config.model)

    models = {}
    optimizers = {}
    schedulers = {}
    ckpt_paths = {} 

    for model_name, model_config in config.model.items():
        if model_config is not None and "params" in model_config and "ckpt_path" in model_config.params:
            ckpt_paths[model_name] = model_config.params.ckpt_path
        else:
            ckpt_paths[model_name] = None 

    for modal_name, ckpt_path in ckpt_paths.items():
        model = Model.load_from_ckpt(ckpt_path)
        models[modal_name] = model

        optimizer_name = f"optimizer_{modal_name}"
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizers[optimizer_name] = optimizer

        scheduler_name = f"scheduler_{modal_name}"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_config.epochs)
        schedulers[scheduler_name] = scheduler

        print(f"Model {modal_name} loaded from {ckpt_path}")
        print(f"Optimizer {optimizer_name} and Scheduler {scheduler_name} initize")


    #### Begin training 
    for epoch in range(epochs):
        for modal_name, model in models.items():
            for param in model.parameters():
                param.requires_grad = False
        for i, data in enumerate(train_dataloader):
            keys_list = list(models.keys())
            random_index = random.randint(0, len(keys_list) - 1)  
            selected_model_name = keys_list[random_index] 
            print(f'{selected_model_name} start to train')
            for param in models[selected_model_name].parameters():
                param.requires_grad = True  

            data = data.to(device)
            
            optimizers[f"optimizer_{selected_model_name}"].zero_grad()
            from models.reconmodels.autoencoder.modules.losses.jointcontrastiveloss import JointContrastiveLoss
            loss = training_config.contrast_weight * JointContrastiveLoss(data) + models[selected_model_name].calculate_loss(data[:, random_index, :, :, :])
            loss.backward()
            optimizers[f"optimizer_{selected_model_name}"].step()

            for param in models[selected_model_name].parameters():
                param.requires_grad = False

            if (epoch+1) % test_epoch == 0:
                pass
            
            if (epoch+1) % save_epoch == 0:
                pass


