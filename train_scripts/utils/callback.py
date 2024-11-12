import os, time, sys, io
from datetime import datetime
import logging
from omegaconf import OmegaConf
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.utils import save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            if trainer.strategy.global_rank == 0:
                start_time = datetime.now()
                print(f"Summoning checkpoint saving in {self.ckptdir}")
                ckpt_path = os.path.join(self.ckptdir, "last_state_dict.ckpt")
                torch.cuda.empty_cache()
                model_state_dict = trainer.model.state_dict()
                torch.save(model_state_dict, ckpt_path)
                # trainer.save_checkpoint(ckpt_path)
                print(f'save ckpt done! use time: {(datetime.now() - start_time).total_seconds() / 60:.2f} minutes')
            print("Exiting program after saving checkpoint.")
            os._exit(0)

    def on_fit_start(self, trainer, pl_module):
        if trainer.strategy.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        reserved_memory = torch.cuda.memory_reserved(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.strategy.reduce(max_memory)
            epoch_time = trainer.strategy.reduce(epoch_time)
            
            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
            rank_zero_info(f"Average Reserved memory {reserved_memory:.2f}MiB")
        except AttributeError as e:
            pass

class ImageLogger(Callback):
    # see https://github.com/CompVis/stable-diffusion/blob/main/main.py
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.tensorboard.TensorBoardLogger: self._log_images_tensorboard
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _log_images_tensorboard(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    # def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not self.disabled and pl_module.global_step > 0:
        #     self.log_img(pl_module, batch, batch_idx, split="val")
        # if hasattr(pl_module, 'calibrate_grad_norm'):
        #     if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                # self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

class GlobalLoggingCallback(Callback):
    def __init__(self, logdir, log_filename='trainer_log.log'):
        super().__init__()
        self.log_dir = logdir
        self.log_filename = log_filename

    def on_fit_start(self, trainer, pl_module):
        if trainer.strategy.global_rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            trainer.strategy.barrier()
        else:
            trainer.strategy.barrier()
        
        log_dir = trainer.strategy.broadcast(self.log_dir, src=0)
        self.log_dir = log_dir
        log_path = os.path.join(log_dir, self.log_filename)
            
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - [Rank %(rank)d] - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )

        current_rank = trainer.strategy.global_rank
        self.logger = logging.LoggerAdapter(logging.getLogger(), {'rank': current_rank})

        sys.stdout = self.StreamToLogger(sys.stdout, self.logger, logging.INFO)
        sys.stderr = self.StreamToLogger(sys.stderr, self.logger, logging.ERROR)

        if trainer.strategy.global_rank == 0:
            self.logger.info(f"Logging to {log_path}")
            self.logger.info("Training started - all output will be logged.")

    def on_fit_end(self, trainer, pl_module):
        self.logger.info("Training finished.")
        # 恢复 sys.stdout 和 sys.stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    class StreamToLogger(object):
        def __init__(self, orgin_stream, logger, log_level=logging.INFO):
            self.orgin_stream = orgin_stream
            self.logger = logger
            self.log_level = log_level
            self.line_buffer = ""

        def write(self, message):
            if message and message.strip() != "":
                if '\r' in message:
                    message = message.replace('\r', '')
                self.logger.log(self.log_level, message.strip())
                self.orgin_stream.write(message)
                self.orgin_stream.flush()

        def flush(self):
            pass  # 不需要实际刷新操作

class ImageSaveCallback(Callback):
    def __init__(self, logdir, log_key='images'):
        super().__init__()
        self.save_dir = logdir + f"/{log_key}"
        self.log_key = log_key
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_epoch_start(self, trainer, pl_module):
        # 重置标记，每个 epoch 开始时重新保存第一个 batch
        self.saved_first_batch = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 仅处理每个 epoch 的第一个 batch
        if not self.saved_first_batch and batch_idx == 0:
            # 获取输入数据、重构数据和潜在空间
            x = pl_module.get_input(batch)  # 获取原图
            recon_x, mu, logvar = pl_module(x)  # 获取重构图和潜在空间
    
            trainer.logger.experiment.add_images(f'{self.log_key}/original', x, trainer.current_epoch, dataformats='NCHW')
            trainer.logger.experiment.add_images(f'{self.log_key}/reconstructed', recon_x, trainer.current_epoch, dataformats='NCHW')
            
            for i, (orig, recon) in enumerate(zip(x, recon_x)):
                save_image(orig, os.path.join(self.save_dir, f"original_epoch_{trainer.current_epoch}_img_{i}.png"))
                save_image(recon, os.path.join(self.save_dir, f"reconstructed_epoch_{trainer.current_epoch}_img_{i}.png"))
            
            self.saved_first_batch = True

class SolarImageLogger(Callback):
    # see original https://github.com/CompVis/stable-diffusion/blob/main/main.py
    def __init__(self, batch_frequency, max_images, clamp=False, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.tensorboard.TensorBoardLogger: self._log_images_tensorboard
        }
        self.log_steps = [200 * n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    def get_cmap_and_limits(self, inputs, mode):
        cmap = "RdBu_r" if mode == 'hmi_image_vae' or mode == 'hmi_image_cliptoken' or mode == 'hmi_image' else "Reds"
        vmin = np.min(inputs)
        vmax = np.max(inputs)
        if mode == 'hmi_image_vae' or mode == 'hmi_image_cliptoken' or mode == 'hmi_image':
            vmax = np.max([np.abs(vmin), np.abs(vmax)]) / 2
            vmin = -vmax
        elif mode == '0094_image' or mode == 'aia0094_image' or mode == 'aia0094_image_vae' or mode == 'aia0094_image_cliptoken_decodelrimage' or mode == 'aia0094_image_cliptoken':  # '0094_image'
            vmax = np.max([np.abs(vmin), np.abs(vmax)]) / 2
            vmin = 0
        else:
            raise ValueError("Unknown modal type")
        return cmap, vmin, vmax

    def get_target_keys(self, pl_module):
        if pl_module.__class__.__name__ in ["SolarLatentDiffusion", "LatentDiffusion"]:
            target_keys = ['inputs', 'inputs_latent', 'reconstruction', 'conditioning', 'conditioning_latent', 'samples', 'samples_latent', 'diffusion_row', 'denoise_row']
        elif pl_module.__class__.__name__ in ["CNN_VAE", "aia0094_CNN_VAE"]:
            target_keys = ['inputs', 'recon', 'mu', 'samples']
        elif pl_module.__class__.__name__ in ["VQModel"]:
            target_keys = ["inputs", 'recon']
        elif pl_module.__class__.__name__ in ["ClipVitDecoder", "SolarLatentGPT", "vit_regressor", "SolarCLIPDAE"]:
            target_keys = ['inputs', 'targets', 'targets_hat']
        elif pl_module.__class__.__name__ in ["ViTMAE"]:
            target_keys = ['inputs', 'targets', 'targets_hat_mask_ratio_set', 'targets_hat_mask_ratio_0', 'targets_hat_inference']
        else:
            raise ValueError("Unsupported model type")
        return target_keys
    
    @rank_zero_only
    def _log_images(self, pl_module, images, modals, batch_idx, split, save_dir=None):
        """
        Logs images either to TensorBoard or saves them locally based on the provided save_dir.
        If save_dir is provided, saves images locally. Otherwise, logs them to TensorBoard.

        Args:
            pl_module: The Lightning module containing the logger.
            images (dict): Dictionary of image tensors.
            modals (dict): Dictionary of modal types for each image key.
            batch_idx (int): Batch index for naming consistency.
            split (str): Split name (train/val/test).
            save_dir (str): Directory to save images locally. If None, logs to TensorBoard.
        """
        target_keys = self.get_target_keys(pl_module)

        for k, img_tensor in images.items():
            if k not in target_keys:
                print(f"Warning: No modal type provided for {k}. Skipping.")
                continue

            image_array = img_tensor.cpu().numpy()
            modal = modals.get(k, None)
            cmap, vmin, vmax = self.get_cmap_and_limits(image_array, modal)
            
            plt.figure(figsize=(32, 16))
            num_images = min(image_array.shape[0], 2)
            for i in range(num_images):
                plt.subplot(1, 2, i+1)
                if len(image_array.shape) == 4:
                    plt.imshow(image_array[i, 0, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
                elif len(image_array.shape) == 3:
                    plt.imshow(image_array[0, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
                plt.title(f"{k} - Image {i}")
                plt.subplots_adjust(wspace=0, hspace=0)

            if save_dir:
                # Save locally
                root = os.path.join(save_dir, "images", split)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, pl_module.global_step, pl_module.current_epoch, batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                plt.savefig(path)
            else:
                # Log to TensorBoard
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plt.close()

                img_rgb = plt.imread(buf)[:, :, :3]
                tag = f"{split}/{k}"
                pl_module.logger.experiment.add_image(
                    tag, img_rgb,
                    global_step=pl_module.global_step, dataformats='HWC'
                )

            plt.close()
        
    @rank_zero_only
    def _log_images_tensorboard(self, pl_module, images, modals, batch_idx, split):
        self._log_images(pl_module, images, modals, batch_idx, split, save_dir=False)

    @rank_zero_only
    def log_local(self, save_dir, split, images, modals, batch_idx, pl_module):
        self._log_images(pl_module, images, modals, batch_idx, split, save_dir=save_dir)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images, modals = pl_module.log_images(batch, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images, modals,
                        batch_idx, pl_module)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, modals, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

