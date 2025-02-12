# 2025/02/12 Suppose all 11 modals can be loaded in a single machine and add ddp training
import math
import sys
import os
import random
from omegaconf import OmegaConf
import argparse
import datetime
import io
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
print(f"current dir {current_dir}")
print(f"parent_dir {parent_dir}")
print(f"cwd {os.getcwd()}")
sys.path.insert(0, parent_dir)
project_root = os.path.abspath(os.path.join(current_dir, "../../../../.."))
sys.path.insert(0, project_root)
from util import instantiate_from_config
from modules.losses.jointcontrastiveloss import JointContrastiveLoss

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
        help="resume traning from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="*",
        metavar="",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=['configs/train_configs/reconmodels/autoencoder/jointvae/JointVAE.yaml']
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
        default="logs/reconmodels/autoencoder/jointvae",
        help="log directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    return parser

def train(rank, world_size, config, opt):
    """ rank: 当前进程的 GPU ID, world_size: 总共的 GPU 数量 """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

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
    train_sampler = DistributedSampler(data.datasets["train"], num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(data.datasets["validation"], num_replicas=world_size, rank=rank, shuffle=False)
    train_dataloader = DataLoader(data.datasets["train"], batch_size=data.batch_size, sampler=train_sampler, num_workers=data.num_workers)
    val_dataloader = DataLoader(data.datasets["validation"], batch_size=data.batch_size, sampler=val_sampler, num_workers=data.num_workers)

    # train_dataloader = DataLoader(data.datasets["train"], batch_size=data.batch_size, 
    #                       shuffle=True, num_workers=data.num_workers)
    # val_dataloader = DataLoader(data.datasets["validation"], batch_size=data.batch_size, 
    #                       shuffle=False, num_workers=data.num_workers)
    for i, data in enumerate(train_dataloader):
        print(data.shape)
        break


    #### Init basic traing config
    training_config = config.training

    # device = training_config.device
    epochs = training_config.epochs
    test_epoch = epochs//training_config.test_freq
    save_epoch = epochs//training_config.save_freq
    

    #### Init Model
    models = {}
    optimizers = {}
    schedulers = {}
    ckpt_paths = {} 

    for modal_name, model_config in config.model.items():
        if model_config is not None and "params" in model_config and "ckpt_path" in model_config.params:
            ckpt_paths[modal_name] = model_config.params.ckpt_path
        else:
            ckpt_paths[modal_name] = None 

    for modal_name, ckpt_path in ckpt_paths.items():
        model = instantiate_from_config(getattr(config.model, modal_name))
        if ckpt_path is not None:
            model = model.load_from_ckpt(getattr(config.model, modal_name).ckpt_path)
        model = model.to(rank)  
        models[modal_name] = DDP(model, device_ids=[rank])

        optimizer_name = f"optimizer_{modal_name}"
        if getattr(config.model, modal_name).base_learning_optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = getattr(config.model, modal_name).base_learning_rate)
        elif getattr(config.model, modal_name).base_learning_optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr = getattr(config.model, modal_name).base_learning_rate)
        else:
            raise ValueError(f'Optimizer {getattr(config.model, modal_name).base_learning_optimizer} is not supported')
        optimizers[optimizer_name] = optimizer

        scheduler_name = f"scheduler_{modal_name}"
        if getattr(config.model, modal_name).base_learning_schedule == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_config.epochs)
        else:
            raise ValueError(f'Scheduler {getattr(config.model, modal_name).base_learning_schedule} is not supported')
        schedulers[scheduler_name] = scheduler

        print(f"Model {modal_name} loaded from {ckpt_path}")
        print(f"Optimizer {optimizer_name} and Scheduler {scheduler_name} initize")

    
    #### Init Logger
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    cfg_fname = os.path.split(opt.config[0])[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    name = cfg_name
    nowname = now + "_" + name + opt.postfix
    logdir = os.path.join(opt.logdir, nowname)   # logs/reconmodels/autoencoder/jointvae/{nowname}/
    ckptdir = os.path.join(logdir, 'checkpoints')
    cfgdir = os.path.join(logdir, 'configs')
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)

    print("Project config")
    print(OmegaConf.to_yaml(config))
    OmegaConf.save(config, os.path.join(cfgdir, "{}-project.yaml".format(now)))

    writer =  SummaryWriter(log_dir = logdir)


    #### Begin training 
    print('Start training')
    gs = 0
    gs_val = 0
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        for modal_name, model in models.items():
            model.module.eval()
            # model.train()
            # for param in model.parameters():
            #     param.requires_grad = False

        loop_train = tqdm(train_dataloader, leave=True)  
        loop_train.set_description(f"Traning Epoch [{epoch+1}/{epochs}]")
        for batch_idx, data in enumerate(loop_train):
            data = data.to(rank)
            keys_list = list(models.keys())
            random_index = random.randint(0, len(keys_list) - 1)  
            selected_model_name = keys_list[random_index]
            # selected_model_name = random.choice(keys_list)
            print(f'Modal {selected_model_name} VAE Model start to train')

            for param in models[selected_model_name].module.parameters():
                param.requires_grad = True

            # data = data.to(device)
            optimizers[f"optimizer_{selected_model_name}"].zero_grad()

            rec_loss, kl_loss = models[selected_model_name].module.calculate_loss(data[:, random_index, :, :, :])
            cor_loss = JointContrastiveLoss(models, data)
            loss = training_config.contrast_weight *cor_loss + training_config.reconstruct_weight*rec_loss + training_config.kl_weight*kl_loss
            print(f'{selected_model_name} loss: {loss}')
            loss.backward()
            optimizers[f"optimizer_{selected_model_name}"].step()

            for param in models[selected_model_name].module.parameters():
                param.requires_grad = False

            if rank == 0:
                loss_dict = {
                    "loss": loss.detach(),
                    "rec_loss": rec_loss.detach(),
                    "kl_loss": kl_loss.detach(),
                    "cor_loss": cor_loss.detach()
                }
                for loss_name, loss_value in loss_dict.items():
                    writer.add_scalar(f'{selected_model_name}/train/{loss_name}', loss_value, gs)
                loop_train.set_postfix(modal=selected_model_name, loss=loss.item(), rec_loss=rec_loss.item(), kl_loss=kl_loss.item(), cor_loss=cor_loss.item())
            gs += 1
            # print(f'one iteration done!')

        schedulers[f"scheduler_{selected_model_name}"].step()
        # print(f'one epoch done!')

        if (epoch+1) % test_epoch == 0:
            print("begin to test")
            loop_test = tqdm(val_dataloader, leave=True)  
            loop_test.set_description(f"Testing Epoch [{epoch+1}/{epochs}]")
            for batch_idx, data in enumerate(loop_test):
                data = data.to(rank)
                for k, (modal_name, model) in enumerate(models.items()):
                    with torch.no_grad():
                        model.module.eval()
                    
                    for j in range(len(keys_list)-1):
                        rec_loss_test, kl_loss_test = models[keys_list[j]].module.calculate_loss(data[:, j, :, :, :])
                        cor_loss_test = JointContrastiveLoss(models, data)
                        loss_test = training_config.contrast_weight *cor_loss + training_config.reconstruct_weight*rec_loss + training_config.kl_weight*kl_loss
                        
                        if rank == 0:
                            loss_dict_test = {
                                "loss": loss_test.detach(),
                                "rec_loss": rec_loss_test.detach(),
                                "kl_loss": kl_loss_test.detach(),
                                "cor_loss": cor_loss_test.detach()
                            }
                            for loss_name, loss_value in loss_dict_test.items():
                                writer.add_scalar(f'{keys_list[j]}/test/{loss_name}', loss_value, gs_val)
                            loop_test.set_postfix(modal=modal_name, loss=loss.item(), rec_loss=rec_loss.item(), kl_loss=kl_loss.item(), cor_loss=cor_loss.item())

                            # painting function #TODO
                            images = {
                                "input": data[:, j, :, :, :],
                                "recon": model.encode(data[:, j, :, :, :])[0]
                            }
                            for image_type, img_tensor in images:
                                image_array = img_tensor.cpu().numpy()
                                cmap = "RdBu_r"
                                vmin = np.min(data)
                                vmax = np.max(data)
                                vmax = np.max([np.abs(vmin), np.abs(vmax)]) / 2
                                vmin = 0
                                # cmap, vmin, vmax = self.get_cmap_and_limits(image_array, modal)
                                
                                plt.figure(figsize=(32, 16))
                                num_images = min(image_array.shape[0], 4)
                                for i in range(num_images):
                                    plt.subplot(1, 2, i+1)
                                    if len(image_array.shape) == 4:
                                        plt.imshow(image_array[i, 0, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
                                    elif len(image_array.shape) == 3:
                                        plt.imshow(image_array[0, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
                                    plt.title(f"{k} - Image {i}")
                                    plt.subplots_adjust(wspace=0, hspace=0)


                                if training_config.img_local: # TODO add img_local bool value can be read by OmegaConf
                                    # Save locally
                                    root = os.path.join(logdir, "images", f'{keys_list[j]}', 'val')
                                    filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(image_type, gs_val, epoch, batch_idx) # TODO add image_type {"inputs", "recon"}
                                    path = os.path.join(root, filename)
                                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                                    plt.savefig(path)
                                else:
                                    buf = io.BytesIO()
                                    plt.savefig(buf, format='png')
                                    buf.seek(0)
                                    plt.close()

                                    img_rgb = plt.imread(buf)[:, :, :3]
                                    tag = f"val/{image_type}"
                                    writer.add_image(
                                        tag, img_rgb,
                                        global_step=gs_val, dataformats='HWC'
                                    )

                    gs_val += 1


        if (epoch+1) % save_epoch == 0:
            print("begin to save")
            for modal_name, model in models.items():
                torch.save({'model': model.module.state_dict(), 'optimizer': optimizers[f"optimizer_{modal_name}"].state_dict(),
                        'scheduler': schedulers[f"scheduler_{modal_name}"].state_dict(), 'epoch': epoch},
                       f'{ckptdir}/{modal_name}/epoch_{epoch+1}.pt')

    writer.close()

if __name__ == "__main__":
    # ckpt_paths should have same order with data make config yaml can be load from cli.
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    config = OmegaConf.load(opt.config[0]) 

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, config, opt), nprocs=world_size)
    
    
    

   