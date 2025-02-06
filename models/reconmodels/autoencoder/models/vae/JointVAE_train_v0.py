import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.reconmodels.autoencoder.util import config_optimizers
from models.reconmodels.autoencoder.util import instantiate_from_config


def main(ddconfig, ckpt_dict, args):
    device = args.device
    train_dataloader = ''

    Model = instantiate_from_config(ddconfig)

    models = {}
    optimizers = {}
    schedulers = {}

    for modal_name, ckpt_path in ckpt_dict.items():
        model = Model.load_from_ckpt(ckpt_path)
        models[modal_name] = model

        optimizer_name = f"optimizer_{modal_name}"
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizers[optimizer_name] = optimizer

        scheduler_name = f"scheduler_{modal_name}"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
        schedulers[scheduler_name] = scheduler

        print(f"Model {modal_name} loaded from {ckpt_path}")
        print(f"Optimizer {optimizer_name} and Scheduler {scheduler_name} initize")

    num_devices = args.num_devices
    device_list = [torch.device(f"cuda:{i}") for i in range(1, num_devices)]  
    central_device = torch.device(f"cuda:0")

    for i, (modal_name, model) in enumerate(models.items()):
        device_index = i % (num_devices - 1)
        model.to(device_list[device_index])  
        optimizers[f"optimizer_{modal_name}"].to(device_list[device_index])
        schedulers[f"scheduler_{modal_name}"].to(device_list[device_index])

    epochs = args.epochs
    test_epoch = epochs//args.test_freq
    save_epoch = epochs//args.save_freq

    for epoch in range(epochs):
        for modal_name, model in models.items():
            for param in model.parameters():
                param.requires_grad = False
        for i, data in enumerate(train_dataloader):
            selected_model_name = random.choice(list(models.keys()))
            for param in models[selected_model_name].parameters():
                param.requires_grad = True  

            data = data.to(device)
            
            optimizers[f"optimizer_{selected_model_name}"].zero_grad()
            from models.reconmodels.autoencoder.modules.losses.jointcontrastiveloss import JointContrastiveLoss
            loss = args.lamda * JointContrastiveLoss(data) + modal_name.calculate_loss(data)
            loss.backward()
            optimizers[f"optimizer_{selected_model_name}"].step()

            for param in models[selected_model_name].parameters():
                param.requires_grad = False

            if (epoch+1) % test_epoch == 0:
                pass



