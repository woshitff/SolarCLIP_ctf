import argparse
import datetime, os, sys, glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

# from pytorch_lightning.utilities.seed import seed_everything
from models.reconmodels.autoencoder.util import instantiate_from_config

def get_parser(**parser_kwargs):

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
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
        "-b",
        "--base",
        nargs="*",
        metavar="/home/chaitf/桌面/SolarCLIP/SolarCLIP_v2/configs/train_configs/reconmodels/ldm/test.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=["/home/chaitf/桌面/SolarCLIP/SolarCLIP_v2/configs/train_configs/reconmodels/autoencoder/vae/test.yaml"],
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
        default="logs",
        help="log directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    return parser

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    sys.path.append(os.getcwd())

    parser = get_parser()
    # parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            model_name = os.path.split(opt.base[0])[0].split("/")[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + model_name + "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)

        # init device
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "devices" in trainer_config:
            cpu = True
            if 'strategy' in trainer_config:
                del trainer_config['strategy']
            trainer_config['accelerator'] = 'cpu'
            trainer_config['devices'] = "auto"
        else:
            cpu = False
            trainer_config['accelerator'] = 'gpu'
            if (isinstance(trainer_config.devices, int) and trainer_config.devices > 1) or \
                (isinstance(trainer_config['devices'], list) and len(trainer_config['devices']) > 1): # use ddp as default
                trainer_config['strategy'] = 'ddp' if not trainer_config.get('strategy', None) else trainer_config['strategy']

        # init callbacks
        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "train.utils.callback.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            # "image_logger": {
            #     "target": "train.utils.callback.ImageLogger",
            #     "params": {
            #         "batch_frequency": 750,
            #         "max_images": 4,
            #         "clamp": True
            #     }
            # },
            # "learning_rate_logger": {
            #     "target": "train.utils.callback.LearningRateMonitor",
            #     "params": {
            #         "logging_interval": "step",
            #         # "log_momentum": True
            #     }
            # },
            "cuda_callback": {
                "target": "train.utils.callback.CUDACallback"
            },
        }
        # if version.parse(pl.__version__) >= version.parse('1.4.0'):
        #     default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 10000,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

        callbacks_kwargs = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
            
        # init model
        model = instantiate_from_config(config.model)
        print(model.device)
        model.learning_rate, model.learning_optimizer, model.learning_schedule = config.model.base_learning_rate, config.model.base_learning_optimizer, config.model.base_learning_schedule
        print(f"Setting learning rate to {model.learning_rate:.2e}")
        print(f"Setting learning optimizer to {model.learning_optimizer}")
        print(f"Setting learning schedule to {model.learning_schedule}")
    
        # init data
        batch_size = 2
        input_size = (batch_size, 1, 1024, 1024)  # 模拟输入的大小

        # 随机生成图像和标签（这里假设 label 是全 1 的图像）
        images = torch.randn(input_size)
        labels = torch.ones(input_size)

        # 创建 TensorDataset 和 DataLoader
        dataset = TensorDataset(images, labels)
        train_dataloader = DataLoader(dataset, batch_size=batch_size)

        tb_logger = TensorBoardLogger(save_dir=logdir, name="tensorboard")
        trainer = Trainer(max_epochs=100, logger=tb_logger, callbacks=callbacks_kwargs)

        

        # Step 4: 运行训练
        try:
            trainer.fit(model, train_dataloader)
        except Exception as e:
            print(f"Training error: {e}")

    except Exception as e:
        print(e)
        print("Failed to parse config files. Please check your syntax.")
