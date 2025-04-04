import os
import argparse
import importlib
from omegaconf import OmegaConf
from packaging import version

import pytorch_lightning as pl
import torch

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

class TrainerSetup:
    def __init__(self, config, lightning_config, trainer_config, opt, now, logdir, cfgdir, ckptdir, model):
        self.config = config
        self.lightning_config = lightning_config
        self.opt = opt
        self.now = now
        self.logdir = logdir
        self.cfgdir = cfgdir
        self.ckptdir = ckptdir
        self.model = model
        self.trainer_config = dict()
        self.trainer_kwargs = dict()

        self.init_trainer_config(trainer_config)
        self.trainer_opt = argparse.Namespace(**self.trainer_config)
        self.lightning_config.trainer = self.trainer_config
        self.init_trainer_kwargs()

    def init_trainer_config(self, trainer_config=None):
        if not "devices" in trainer_config:
            if 'strategy' in trainer_config:
                del trainer_config['strategy']
            trainer_config['accelerator'] = 'gpu' if torch.cuda.is_available() else 'cpu'
            trainer_config['devices'] = "auto"
        else:
            trainer_config['accelerator'] = 'gpu'
            if (isinstance(trainer_config.devices, int) and trainer_config.devices > 1) or \
                (isinstance(trainer_config['devices'], list) and len(trainer_config['devices']) > 1): # use ddp as default
                trainer_config['strategy'] = trainer_config.get('strategy', 'ddp_find_unused_parameters_true')

        self.trainer_config = trainer_config

            
    def init_trainer_kwargs(self):
        self._init_logger()
        self._init_callbacks()

    def _init_logger(self):
        default_logger_cfgs = {
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "name": "tensorboard",
                    "save_dir": self.logdir,
                }
            }
        }
        default_logger_cfg = default_logger_cfgs["tensorboard"]
        if "logger" in self.lightning_config:
            logger_cfg = self.lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        self.trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    def _init_callbacks(self):
        default_callbacks_cfg = {
            # "global_logging": {
            #     "target": "train_scripts.utils.callback.GlobalLoggingCallback",
            #     "params": {
            #         "logdir": self.logdir,
            #     }
            # },
            "setup_callback": {
                "target": "train_scripts.utils.callback.SetupCallback",
                "params": {
                    "resume": self.opt.resume,
                    "now": self.now,
                    "logdir": self.logdir,
                    "ckptdir": self.ckptdir,
                    "cfgdir": self.cfgdir,
                    "config": self.config,
                    "lightning_config": self.lightning_config,
                }
            },
            "image_logger": {
                "target": "train_scripts.utils.callback.SolarImageLogger",
                "params": {
                    "batch_frequency": 75000,
                    # "batch_frequency": 1,
                    "max_images": 4,
                    "clamp": False,
                }
            },
            "learning_rate_logger": {
                "target": "train_scripts.utils.callback.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "train_scripts.utils.callback.CUDACallback"
            }
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': self._init_checkpoint_callback()})
        if "callbacks" in self.lightning_config:
            callbacks_cfg = self.lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()
        # if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
        if True:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(self.ckptdir, 'trainstep_checkpoints'),
                         "filename": "best_val_loss_epoch",
                         "verbose": True,
                         'save_top_k': 1,
                         'monitor':'val/loss_epoch',
                         'every_n_epochs': 1,
                         'save_weights_only': True
                     }
                     }
            }
            # early_stop_callbacks_dict = {
            #     'early_stop':
            #     {'target':'pytorch_lightning.callbacks.EarlyStopping',
            #      'params':{
            #             'monitor':"val/loss_epoch",   
            #             'patience':30,           
            #             'verbose':True,         
            #             'mode':"min"  
            #     }
            #     }
            # }
            if 'default_metrics_over_trainsteps_ckpt' not in callbacks_cfg:
                default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)
            # if 'early_stop' not in callbacks_cfg:
            #     default_callbacks_cfg.update(early_stop_callbacks_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(self.trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = self.trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        self.trainer_kwargs["callbacks"] = [instantiate_from_config(cfg) for cfg in callbacks_cfg.values()]

    def _init_checkpoint_callback(self):
        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": self.ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": False,
                'save_top_k': -1,
                'every_n_epochs': 20,
                'save_weights_only': True
            }
        }
        # if hasattr(self.model, "monitor"):
        #     print(f"Monitoring {self.model.monitor} as checkpoint metric.")
        #     default_modelckpt_cfg["params"]["monitor"] = self.model.monitor
        #     default_modelckpt_cfg["params"]["save_top_k"] = 3
        if "modelcheckpoint" in self.lightning_config:
            modelckpt_cfg = self.lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            self.trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        return modelckpt_cfg
    

