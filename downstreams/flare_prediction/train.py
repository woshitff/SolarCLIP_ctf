import math
import sys
import os
import random
from omegaconf import OmegaConf
import argparse
import datetime


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

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
from .utils import instantiate_from_config
from models.reconmodels.autoencoder.models.vae.CNN_VAE_v2 import CNN_VAE_two as base_model


if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    if "A40" in gpu_name or "A100" in gpu_name:
        torch.set_float32_matmul_precision('high') # highest, high, medium
        print(f'device is {gpu_name}, set float32_matmul_precision to high')
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
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        help="pytorch lightning logger type",
    )

    return parser

class classify_model(base_model):
    '''
    A wrapper class to add classifier to the model.
    '''
    def __init__(self, num_class, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = 512
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_class)
        )
    def classify(self, x):
        '''
        x: (b, c, h, w)
        return: (b, num_class)
        '''
        x = self.encode(x)[0]
        x = self.get_logit(x) # (b, c)
        x = self.classifier(x) # (b, num_class)
        return x

class multi_model(pl.LightningModule):
    """
    A wrapper class for vae model to be used in a single training loop.
    """
    def __init__(self, config:OmegaConf):
        super(classify_model, self).__init__()
        self.config = config
        self.get_models()

    # funcitons for initialization
    def get_models(self):
        """
        Instantiate the models specified in the config.
        """
        self.models = nn.ModuleDict()
        self.modal_to_id = {}
        self.id_to_modal = {}
        for i, (modal_name, model_config) in enumerate(self.config.model.items()):
            self.modal_to_id[modal_name] = i
            self.id_to_modal[i] = modal_name
            model = instantiate_from_config(model_config)
            self.models[modal_name] = model
            print(f"Model {modal_name} loaded from {model_config.params.ckpt_path}")

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        for i in range(len(self.models)):
            modal_name = self.id_to_modal[i]
            if getattr(self.config.model, modal_name).base_learning_optimizer == 'Adam':
                optimizer = torch.optim.Adam(self.models[modal_name].parameters(), lr = getattr(self.config.model, modal_name).base_learning_rate)
            elif getattr(self.config.model, modal_name).base_learning_optimizer == 'AdamW':
                optimizer = torch.optim.AdamW(self.models[modal_name].parameters(), lr = getattr(self.config.model, modal_name).base_learning_rate)
            else:
                raise ValueError(f'Optimizer {getattr(self.config.model, modal_name).base_learning_optimizer} is not supported')
            optimizers.append(optimizer)

            if getattr(self.config.model, modal_name).base_learning_schedule == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.training.epochs)
            else:
                raise ValueError(f'Scheduler {getattr(self.config.model, modal_name).base_learning_schedule} is not supported')
            schedulers.append(scheduler)
        return optimizers, schedulers
    
    # functions for training
    def training_step(self, batch, batch_idx):
        current_epoch = self.current_epoch
        contrast_weight = self.config.training.contrast_weight_min + (self.config.training.contrast_weight_max - self.config.training.contrast_weight_min) * math.sin(math.pi/2 * current_epoch / self.config.training.epochs) # increase contrast weight from min to max
        label = torch.arange(batch.shape[0]).to(batch.device)  # (b,) label for contrastive loss

        if self.full_model_train:
            rec_loss_ = {}
            kld_loss_ = {}
            logits_ = {}
            contrast_loss_ = {}
            for name, model in self.models.items():
                # get rec_loss, kld_loss and logits for each model
                model.train()
                rec_loss, kld_loss, mu, _, _ = model.calculate_loss(batch[:, self.data_modal_to_id[name], :, :, :], return_moment=True)
                rec_loss_[name] = rec_loss
                kld_loss_[name] = kld_loss
                logit = model.get_logit(mu)
                # logit = model.class_block(mu)  # (b, c)
                logit = logit/(logit.norm(dim=1, keepdim=True)+ 1e-32)  # (b, c)
                logits_[name] = logit

            if self.mean_logit:
                mean_logit = torch.mean(torch.stack(list(logits_.values())), dim=0)  # (b, c)
                mean_logit = mean_logit/(mean_logit.norm(dim=1, keepdim=True)+ 1e-32)  # (b, c)
            
            # calculate contrast loss
            for name, model in self.models.items():
                contrast_loss = 0
                if self.mean_logit:
                    cor_matrix = torch.matmul(logits_[name], mean_logit.T)
                    contrast_loss = F.cross_entropy(cor_matrix, label)
                    contrast_loss_[name] = contrast_loss
                else:
                    for name2, model2 in self.models.items():
                        if name != name2:
                            cor_matrix = torch.matmul(logits_[name], logits_[name2].T)
                            contrast_loss += F.cross_entropy(cor_matrix, label)
                    contrast_loss_[name] = contrast_loss/ (len(self.models)-1) # average contrast loss for each model

            # optimize
            rec_loss = sum(rec_loss_.values())
            kld_loss = sum(kld_loss_.values())
            contrast_loss = sum(contrast_loss_.values())
            loss = contrast_weight * contrast_loss + self.config.training.reconstruct_weight * rec_loss + self.config.training.kl_weight * kld_loss
            optimizers = self.optimizers()
            for optimizer in optimizers:
                optimizer.zero_grad()
            self.manual_backward(loss)
            # check gradients
            # if self.global_rank == 0:
            #     for name, model in self.models.items():
            #         print(f"Model {name} parameters:")
            #         for param_name, param in model.class_block.named_parameters():
            #             print(f"Parameter name: {param_name}")
            #             if param.grad is not None:
            #                 print("Gradient:", param.grad)
            #             else:
            #                 print("Gradient: None")
            #         break     
            for optimizer in optimizers:
                optimizer.step()

            # log
            for name, model in self.models.items():
                self.log(f"{name}/train/loss", loss, logger=True, on_epoch=True, sync_dist=True)
                self.log(f"{name}/train/rec_loss/", rec_loss_[name], logger=True, on_epoch=True, sync_dist=True)
                self.log(f"{name}/train/kld_loss", kld_loss_[name], logger=True, on_epoch=True, sync_dist=True)
                self.log(f"{name}/train/contrast_loss", contrast_loss_[name], logger=True, on_epoch=True, sync_dist=True)
                self.log(f"contrast_weight", contrast_weight, logger=True, on_epoch=True, sync_dist=True)
                self.log(f'{name}/scheduler', self.lr_schedulers()[self.modal_to_id[name]].get_last_lr()[0], logger=True, on_epoch=True, sync_dist=True)
            self.log(f"avg/train/loss", loss/len(self.models), logger=True, on_epoch=True, sync_dist=True)
            self.log(f"avg/train/rec_loss", rec_loss/len(self.models), logger=True, on_epoch=True, sync_dist=True)
            self.log(f"avg/train/kld_loss", kld_loss/len(self.models), logger=True, on_epoch=True, sync_dist=True)
            self.log(f"avg/train/contrast_loss", contrast_loss/len(self.models), logger=True, on_epoch=True, sync_dist=True)

        else:
            training_id = random.randint(0, len(self.models) - 1)  # randomly select a model to train
            training_modal = self.id_to_modal[training_id]  # get the training modal name
            self.models[training_modal].train()  # set the selected model to train mode

            # loss for the selected model
            data_id = self.data_modal_to_id[training_modal]  # get the data id for the selected model
            rec_loss, kld_loss, mu, _, _ = self.models[training_modal].calculate_loss(batch[:, data_id, :, :, :], return_moment=True)

            # contrastive loss
            contrast_loss = 0
            logit = self.models[training_modal].get_logit(mu)  # (b, c)
            logit = logit/(logit.norm(dim=1, keepdim=True)+ 1e-32)  # (b, c)
            other_logits = {}
            for i in range(len(self.models)):
                if i != training_id:
                    compare_modal = self.id_to_modal[i]  # get the compare modal name
                    self.models[compare_modal].eval()  # set the other models to eval mode
                    data_id = self.data_modal_to_id[compare_modal]  # get the data id for the other model
                    with torch.no_grad():
                        other_logit = self.models[compare_modal].get_logit(self.models[compare_modal].encode(batch[:, data_id, :, :, :])[0])  # (b, c)
                        other_logit = other_logit/(other_logit.norm(dim=1, keepdim=True)+ 1e-32)  # (b, c)
                        other_logits[compare_modal] = other_logit
            if self.mean_logit:
                other_logits[training_modal] = logit
                mean_logit = torch.mean(torch.stack(list(other_logits.values())), dim=0)  # (b, c)
                cor_matrix = torch.matmul(logits_[name], mean_logit.T)
                contrast_loss = F.cross_entropy(cor_matrix, label)
            else:
                for other_logit in other_logits.values():
                    cor_matrix = torch.matmul(logit, other_logit.T)  # (b, b)
                    contrast_loss += F.cross_entropy(cor_matrix, label)
                contrast_loss = contrast_loss / (len(self.models) - 1)  # average contrast loss for the selected model
            loss = contrast_weight * contrast_loss + self.config.training.reconstruct_weight * rec_loss + self.config.training.kl_weight * kld_loss

            # optimize
            optimizers = self.optimizers()
            optimizers[training_id].zero_grad()
            self.manual_backward(loss)
            optimizers[training_id].step()

            # log
            self.log(f"{training_modal}/train/loss", loss, logger=True, on_epoch=True, sync_dist=True)
            self.log(f"{training_modal}/train/rec_loss", rec_loss, logger=True, on_epoch=True, sync_dist=True)
            self.log(f"{training_modal}/train/kld_loss", kld_loss, logger=True, on_epoch=True, sync_dist=True)
            self.log(f"{training_modal}/train/contrast_loss", contrast_loss, logger=True, on_epoch=True, sync_dist=True)
            self.log(f"contrast_weight", contrast_weight, logger=True, on_epoch=True, sync_dist=True)
            self.log(f'{training_modal}/scheduler', self.lr_schedulers()[training_id].get_last_lr()[0], logger=True, on_epoch=True, sync_dist=True)
    
    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            for scheduler in schedulers:
                scheduler.step()

    def validation_step(self, batch, batch_idx):
        current_epoch = self.current_epoch
        contrast_weight = self.config.training.contrast_weight_min + (self.config.training.contrast_weight_max - self.config.training.contrast_weight_min) * math.sin(math.pi/2 * current_epoch / self.config.training.epochs) # increase contrast weight from min to max
        label = torch.arange(batch.shape[0]).to(batch.device)  # (b,) label for contrastive loss

        with torch.no_grad():
            rec_loss_ = {}
            kld_loss_ = {}
            logits_ = {}
            contrast_loss_ = {}
            for name, model in self.models.items():
                # get rec_loss, kld_loss and logits for each model
                model.train()
                rec_loss, kld_loss, mu, _, _ = model.calculate_loss(batch[:, self.data_modal_to_id[name], :, :, :], return_moment=True)
                rec_loss_[name] = rec_loss
                kld_loss_[name] = kld_loss
                logit = model.get_logit(mu)
                logit = logit/(logit.norm(dim=1, keepdim=True)+ 1e-32)  # (b, c)
                logits_[name] = logit

            if self.mean_logit:
                mean_logit = torch.mean(torch.stack(list(logits_.values())), dim=0)  # (b, c)
                mean_logit = mean_logit/(mean_logit.norm(dim=1, keepdim=True)+ 1e-32)  # (b, c)
            
            # calculate contrast loss
            for name, model in self.models.items():
                contrast_loss = 0
                if self.mean_logit:
                    cor_matrix = torch.matmul(logits_[name], mean_logit.T)
                    contrast_loss = F.cross_entropy(cor_matrix, label)
                    contrast_loss_[name] = contrast_loss
                else:
                    for name2, model2 in self.models.items():
                        if name != name2:
                            cor_matrix = torch.matmul(logits_[name], logits_[name2].T)
                            contrast_loss += F.cross_entropy(cor_matrix, label)
                    contrast_loss_[name] = contrast_loss/ (len(self.models)-1) # average contrast loss for each model

            # optimize
            rec_loss = sum(rec_loss_.values())
            kld_loss = sum(kld_loss_.values())
            contrast_loss = sum(contrast_loss_.values())
            loss = contrast_weight * contrast_loss + self.config.training.reconstruct_weight * rec_loss + self.config.training.kl_weight * kld_loss

            # log
            for name, model in self.models.items():
                self.log(f"{name}/val/loss", loss, logger=True, on_epoch=True, sync_dist=True)
                self.log(f"{name}/val/rec_loss/", rec_loss_[name], logger=True, on_epoch=True, sync_dist=True)
                self.log(f"{name}/val/kld_loss", kld_loss_[name], logger=True, on_epoch=True, sync_dist=True)
                self.log(f"{name}/val/contrast_loss", contrast_loss_[name], logger=True, on_epoch=True, sync_dist=True)
            self.log(f"avg/val/loss", loss/len(self.models), logger=True, on_epoch=True, sync_dist=True)
            self.log(f"avg/val/rec_loss", rec_loss/len(self.models), logger=True, on_epoch=True, sync_dist=True)
            self.log(f"avg/val/kld_loss", kld_loss/len(self.models), logger=True, on_epoch=True, sync_dist=True)
            self.log(f"avg/val/contrast_loss", contrast_loss/len(self.models), logger=True, on_epoch=True, sync_dist=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default='./config.yaml')
    args = parser.parse_args()

    try:
        config = OmegaConf.load(args.config_dir)
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
    solar_predictor = SolarPredictor(model, predictor, **config.solarpredictor)

    os.makedirs(config.training.log_dir, exist_ok=True)
    os.makedirs(config.training.save_dir, exist_ok=True)

    if config.training.seed:
        print(f'set seed to {config.training.seed}')
        seed_everything(config.training.seed)
    else:
        print('random seed is not set')

    #load data
    if config.data_configs.num_workers <0:
        trainer = Trainer()
        suggested_num_workers = trainer.num_devices*trainer.num_nodes * 2
        config.data_configs.num_workers = suggested_num_workers
    train_dataset = SolarDataset(config.data_configs, mode='train')
    val_dataset = SolarDataset(config.data_configs, mode='eval')
    train_dataloader = DataLoader(train_dataset, batch_size=config.data_configs.batch_size, shuffle=True, num_workers=config.data_configs.num_workers, pin_memory=config.data_configs.pin_memory)
    val_dataloader = DataLoader(val_dataset, batch_size=config.data_configs.batch_size, shuffle=False, num_workers=config.data_configs.num_workers, pin_memory=config.data_configs.pin_memory)


#train----------------------------------------------------------------------------------------------------------------

    # logger
    if config.logger.type == 'wandb':
        from pytorch_lightning.loggers import WandbLogger
        import wandb
        if 'key' in config.logger.config.keys():
            key = config.logger.config.pop('key')
            wandb.login(key=key)


        logger = WandbLogger(**config.logger.config)
    elif config.logger.type == 'tensorboard':
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(config.training.log_dir)
    else:
        logger = None

    # checkpoint
    callbacks = []
    from pytorch_lightning.callbacks import ModelCheckpoint
    if config.training.checkpoint.every_n_train_steps:
        config.training.checkpoint.every_n_epochs = None
    else:
        config.training.checkpoint.every_n_train_steps = None
        if not config.training.checkpoint.every_n_epochs:
            print('Warning: checkpoint step is not given, using default every_n_epochs=1')
            config.training.checkpoint.every_n_epochs = 1
    checkpoint = ModelCheckpoint(
        filename=f'{config.logger.config.name}-'+'{epoch}-{step}-{acc_val_mean:.4f}-{acc_val:.4f}',
        dirpath=config.training.save_dir,
        save_top_k=config.training.checkpoint.save_top_k,
        verbose=True,
        monitor='acc_val_mean',
        mode='max',
        every_n_train_steps= config.training.checkpoint.every_n_train_steps,
        every_n_epochs=config.training.checkpoint.every_n_epochs,
    )
    callbacks.append(checkpoint)

    # early stop
    if config.training.early_stop:
        from pytorch_lightning.callbacks import EarlyStopping
        early_stop = EarlyStopping('val_loss', patience=config.training.early_stop,verbose=True)
        callbacks.append(early_stop)

    OmegaConf.save(config, os.path.join(config.training.log_dir, f'{config.logger.config.name}.yaml'))
    print(OmegaConf.to_yaml(config))

    trainer = Trainer(logger=logger, max_epochs=config.training.epochs, callbacks=callbacks, enable_checkpointing=config.training.enable_checkpointing, precision=config.training.precision, strategy=config.training.strategy, log_every_n_steps=1)
    trainer.fit(solar_predictor, train_dataloader, val_dataloader, ckpt_path=config.training.load_dir)

    print('finish')
