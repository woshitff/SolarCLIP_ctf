# 2025/02/12 Suppose all 11 modals can be loaded in a single machine and add ddp training
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
from util import instantiate_from_config


if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    if "A40" in gpu_name or "A100" in gpu_name:
        torch.set_float32_matmul_precision('high') # highest, high, medium
        print(f'device is {gpu_name}, set float32_matmul_precision to high')

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
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        help="pytorch lightning logger type",
    )

    return parser

class multi_model(pl.LightningModule):
    """
    A wrapper class for multiple models to be used in a single training loop.
    """
    def __init__(self, config:OmegaConf):
        super(multi_model, self).__init__()
        self.config = config
        self.automatic_optimization = False # to use manual optimization
        self.get_models()

        self.data_id_to_modal = self.config.data.params.train.params.modal_list
        assert self.data_id_to_modal == self.config.data.params.validation.params.modal_list # train and val modal list should be the same
        assert len(self.data_id_to_modal) == len(self.models), "train and val modal list should be the same as the model list"
        self.data_modal_to_id = { modal: i for i, modal in enumerate(self.data_id_to_modal) }

    

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

            scheduler_name = f"scheduler_{modal_name}"
            if getattr(self.config.model, modal_name).base_learning_schedule == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.training.epochs)
            else:
                raise ValueError(f'Scheduler {getattr(self.config.model, modal_name).base_learning_schedule} is not supported')
            schedulers.append(scheduler)
        return optimizers, schedulers
    
    # functions for training
    def training_step(self, batch, batch_idx):
        # hyperparameters: contrast_weight, training_id
        current_epoch = self.current_epoch
        contrast_weight = self.config.training.contrast_weight_min + (self.config.training.contrast_weight_max - self.config.training.contrast_weight_min) * math.sin(math.pi/2 * current_epoch / self.config.training.epochs) # increase contrast weight from min to max

        training_id = random.randint(0, len(self.models) - 1)  # randomly select a model to train
        training_modal = self.id_to_modal[training_id]  # get the training modal name
        self.models[training_modal].train()  # set the selected model to train mode

        # loss for the selected model
        data_id = self.data_modal_to_id[training_modal]  # get the data id for the selected model
        rec_loss, kld_loss, mu, _, _ = self.models[training_modal].calculate_loss(batch[:, data_id, :, :, :], return_moment=True)

        # contrastive loss
        contrast_loss = 0
        label = torch.arange(batch.shape[0]).to(batch.device)  # (b,)
        logit = self.models[training_modal].get_logit(mu)  # (b, c)
        logit = logit/(logit.norm(dim=1, keepdim=True)+ 1e-32)  # (b, c)
        for i in range(len(self.models)):
            if i != training_id:
                compare_modal = self.id_to_modal[i]  # get the compare modal name
                self.models[compare_modal].eval()  # set the other models to eval mode
                data_id = self.data_modal_to_id[compare_modal]  # get the data id for the other model
                with torch.no_grad():
                    other_logit = self.models[compare_modal].get_logit(self.models[compare_modal].encode(batch[:, data_id, :, :, :])[0])  # (b, c)
                    other_logit = other_logit/(other_logit.norm(dim=1, keepdim=True)+ 1e-32)  # (b, c)
                    cor_matrix = torch.matmul(logit, other_logit.T)  # (b, b)
                    contrast_loss += F.cross_entropy(cor_matrix, label)
        loss = contrast_weight * contrast_loss + self.config.training.reconstruct_weight * rec_loss + self.config.training.kl_weight * kld_loss

        # optimize
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()
        optimizers[training_id].zero_grad()
        self.manual_backward(loss)
        optimizers[training_id].step()
        schedulers[training_id].step()

        self.log(f"train_loss_{self.id_to_modal[training_id]}", loss, logger=True, on_epoch=True)
        self.log(f"train_rec_loss_{self.id_to_modal[training_id]}", rec_loss, logger=True, on_epoch=True)
        self.log(f"train_kld_loss_{self.id_to_modal[training_id]}", kld_loss, logger=True, on_epoch=True)
        self.log(f"train_contrast_loss_{self.id_to_modal[training_id]}", contrast_loss, logger=True, on_epoch=True)
        self.log(f"contrast_weight", contrast_weight, logger=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        current_epoch = self.current_epoch
        contrast_weight = self.config.training.contrast_weight_min + (self.config.training.contrast_weight_max - self.config.training.contrast_weight_min) * math.sin(math.pi/2 * current_epoch / self.config.training.epochs) # increase contrast weight from min to max

        with torch.no_grad():
            for name, model in self.models.items():
                model.eval()
            training_id = random.randint(0, len(self.models) - 1)
            training_modal = self.id_to_modal[training_id]
            data_id = self.data_modal_to_id[training_modal]
            rec_loss, kld_loss, mu, _, _ = self.models[training_modal].calculate_loss(batch[:, data_id, :, :, :], return_moment=True)
            contrast_loss = 0
            label = torch.arange(batch.shape[0]).to(batch.device)  # (b,)
            logit = self.models[training_modal].get_logit(mu)  # (b, c)
            logit = logit/(logit.norm(dim=1, keepdim=True)+ 1e-32)  # (b, c)
            for i in range(len(self.models)):
                if i != training_id:
                    compare_modal = self.id_to_modal[i]
                    data_id = self.data_modal_to_id[compare_modal]
                    other_logit = self.models[compare_modal].get_logit(self.models[compare_modal].encode(batch[:, data_id, :, :, :])[0])  # (b, c)
                    other_logit = other_logit/(other_logit.norm(dim=1, keepdim=True)+ 1e-32)  # (b, c)
                    cor_matrix = torch.matmul(logit, other_logit.T)  # (b, b)
                    contrast_loss += F.cross_entropy(cor_matrix, label)
            loss = contrast_weight * contrast_loss + self.config.training.reconstruct_weight * rec_loss + self.config.training.kl_weight * kld_loss

            self.log(f"val_loss_{self.id_to_modal[training_id]}", loss, logger=True, on_epoch=True, sync_dist=True)
            self.log(f"val_rec_loss_{self.id_to_modal[training_id]}", rec_loss, logger=True, on_epoch=True, sync_dist=True)
            self.log(f"val_kld_loss_{self.id_to_modal[training_id]}", kld_loss, logger=True, on_epoch=True, sync_dist=True)
            self.log(f"val_contrast_loss_{self.id_to_modal[training_id]}", contrast_loss, logger=True, on_epoch=True, sync_dist=True)
         

def solar_painting(image_array, modal, title = None):
    assert len(image_array.shape) == 4 # (b, c, h, w)
    if modal == "hmi":
        cmap = "RdBu_r"
        vmax = np.max(np.abs(image_array))
        vmin = -vmax
    else:
        cmap = "Reds"
        vmin = 0
        vmax = np.max(image_array)
    fig = plt.figure(figsize=(32, 16))
    num_images = min(image_array.shape[0], 4)
    for i in range(num_images):
        plt.subplot(1, 2, i+1)
        plt.imshow(image_array[i, 0, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.subplots_adjust(wspace=0, hspace=0)
    return fig

def latent_painting(image_array, modal, title = None):
    assert len(image_array.shape) == 4 # (b, c, h, w)
    fig = plt.figure(figsize=(32, 16))
    num_images = min(image_array.shape[0], 4)
    c = image_array.shape[1]
    visual_channels = np.random.choice(c, 3, replace=(c < 3))
    image_array = image_array[:, visual_channels, :, :]
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    for i in range(num_images):
        plt.subplot(1, 2, i+1)
        plt.imshow(image_array[i,: :, :].permute(1, 2, 0))
        plt.title(title)
        plt.subplots_adjust(wspace=0, hspace=0)
    return fig

class MultiCheckpoint(ModelCheckpoint):
    def __init__(self, save_image_local = False, **kwargs):
        super().__init__(**kwargs)
        self.save_image_local =save_image_local

    def _save_model(self, trainer: pl.trainer, pl_module: pl.LightningModule, file_path: str):
        super()._save_model(trainer, pl_module, file_path) # Call the original save model method

        if trainer.is_global_zero:
            # prepare for plotting
            print("Saving model and plotting images...")
            if trainer.val_dataloaders is not None:
                val_loader = trainer.val_dataloaders[0]
                random_batch_idx = torch.randint(0, len(val_loader), (1,)).item()
                data = val_loader.dataset[random_batch_idx]
                data = data.to(trainer.root_gpu)  #  (b, c, h, w)
            else:
                data = None
            print(f"current model to {file_path}")
            for name, model in pl_module.models.items():
                # save multi model checkpoints
                save_path = os.path.join(os.path.dirname(file_path), name)
                os.makedirs(save_path, exist_ok=True)
                print(f"Saving model to {save_path}")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": trainer.optimizers[pl_module.modal_to_id[name]].state_dict(),
                    "scheduler": trainer.lr_schedulers[pl_module.modal_to_id[name]].state_dict(),
                    "epoch": trainer.current_epoch,
                }, os.path.join(save_path, f"epoch_{trainer.current_epoch}.pt"))
                print(f"Saved model to {os.path.join(save_path, f'epoch_{trainer.current_epoch}.pt')}")
                # plot images
                if data is not None:
                    with torch.no_grad():
                        model.eval()
                        mu = model.encode(data[:, pl_module.data_modal_to_id[name], :, :, :])[0]  # (b, c, h, w)
                        rec = model.decode(mu)  # (b, c, h, w)
                    input_fig = solar_painting(data[:, pl_module.data_modal_to_id[name], :, :, :].cpu().numpy(), name, title = f"{name} - Input")
                    rec_fig = solar_painting(rec.cpu().numpy(), name, title = f"{name} - Reconstructed")
                    latent_fig = latent_painting(mu.cpu().numpy(), name, title = f"{name} - Latent")
                    if self.save_image_local:
                        input_fig.savefig(os.path.join(save_path, f"epoch_{trainer.current_epoch+1}_input.png"))
                        rec_fig.savefig(os.path.join(save_path, f"epoch_{trainer.current_epoch+1}_recon.png"))
                        latent_fig.savefig(os.path.join(save_path, f"epoch_{trainer.current_epoch+1}_latent.png"))
                    if trainer.logger:
                        trainer.logger.experiment.add_figure(f"{name}/input", input_fig, global_step=trainer.global_step)
                        trainer.logger.experiment.add_figure(f"{name}/recon", rec_fig, global_step=trainer.global_step)
                        trainer.logger.experiment.add_figure(f"{name}/latent", latent_fig, global_step=trainer.global_step)
                    plt.close(input_fig)
                    plt.close(rec_fig)
                    plt.close(latent_fig)
                print(f"Saved images to {os.path.join(save_path, f'epoch_{trainer.current_epoch}.png')}")
                
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                        


def train(config, opt):

    trainer = pl.Trainer() # to locate rank 0
    pl.seed_everything(opt.seed)

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
    train_dataloader = DataLoader(data.datasets["train"], batch_size=data.batch_size, 
                          shuffle=True, num_workers=data.num_workers)
    val_dataloader = DataLoader(data.datasets["validation"], batch_size=data.batch_size, 
                          shuffle=False, num_workers=data.num_workers)


    #### Init basic traing config
    training_config = config.training

    epochs = training_config.epochs
    test_epoch = epochs//training_config.test_freq
    save_epoch = epochs//training_config.save_freq

    #### Init Logger
    if trainer.is_global_zero:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        cfg_fname = os.path.split(opt.config[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = cfg_name
        nowname = now + "_" + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)   # logs/reconmodels/autoencoder/jointvae/{nowname}/
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    ckptdir = os.path.join(logdir, 'checkpoints')
    cfgdir = os.path.join(logdir, 'configs')
    print(f"Logdir: {logdir}, ckptdir: {ckptdir}, cfgdir: {cfgdir}")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    print("Logdir: ", logdir, "ckptdir: ", ckptdir, "cfgdir: ", cfgdir, '\n 1\n')

    print("Project config")
    # print(OmegaConf.to_yaml(config))
    OmegaConf.save(config, os.path.join(cfgdir, "{}-project.yaml".format(now)))
    print("Logdir: ", logdir, "ckptdir: ", ckptdir, "cfgdir: ", cfgdir,'\n 2\n')

    if opt.logger == 'wandb':
        logger = None # TODO add wandb logger
    elif opt.logger == 'tensorboard':
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir = logdir)
    else:
        logger = None


    #### Init checkpoint
    checkpoint_callback = MultiCheckpoint(
        dirpath=ckptdir,
        filename="{epoch:02d}",
        save_top_k=-1,
        every_n_epochs=save_epoch,
        save_image_local=config.training.img_local,
    )
    print("Logdir: ", logdir, "ckptdir: ", ckptdir, "cfgdir: ", cfgdir,'\n 3\n')

    #### init model

    model = multi_model(config=config)

    #### init trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        # precision='bf16-mixed',
        strategy='ddp_find_unused_parameters_true',
        max_epochs=epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        check_val_every_n_epoch = test_epoch,
    )
    #### training
    print("Logdir: ", logdir, "ckptdir: ", ckptdir, "cfgdir: ", cfgdir,'\n 4\n')
    if opt.resume:
        resume_path = os.path.join(ckptdir, opt.resume)
        if os.path.isdir(resume_path):
            dirs = os.listdir(resume_path)
            dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
            resume_path = os.path.join(resume_path, dirs[-1])
        print(f"Resume from {resume_path}")
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=resume_path)
    else:
        trainer.fit(model, train_dataloader, val_dataloader)




if __name__ == "__main__":
    # ckpt_paths should have same order with data make config yaml can be load from cli.
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    config = OmegaConf.load(opt.config[0]) 

    train(config, opt)
    print("Training finished")
    
    
    

   