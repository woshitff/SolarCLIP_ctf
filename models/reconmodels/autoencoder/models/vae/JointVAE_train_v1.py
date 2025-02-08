# 2025/02/08 Suppose all 11 modals can be loaded in a single machine.
import math
import os
import random
import yaml
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.reconmodels.autoencoder.util import config_optimizers
from models.reconmodels.autoencoder.util import instantiate_from_config

# config
# 1. model config
# 2. data config
# 3. train config
if __name__ == "__main__":
    # ckpt_dict should have same order with data
    config = OmegaConf.load("configs/train_configs/reconmodels/autoencoder/vae/JointVAE.yaml") # TODO make config yaml can be load from cli.


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

    ckpt_dict = {} # TODO

    for modal_name, ckpt_path in ckpt_dict.items():
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


