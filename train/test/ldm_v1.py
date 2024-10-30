import argparse
import datetime, os, sys, glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import seed_everything

from models.reconmodels.autoencoder.util import instantiate_from_config
from train.utils.util import TrainerSetup
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
        default=["/mnt/nas/home/huxing/202407/ctf/SolarCLIP_ctf_v2/SolarCLIP_ctf/configs/train_configs/reconmodels/ldm/hmi2aia0094_test.yaml"],
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
        default=True,
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
        default=False,
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

        #### init model
        model = instantiate_from_config(config.model)
        model.learning_rate, model.learning_optimizer, model.learning_schedule = config.model.base_learning_rate, config.model.base_learning_optimizer, config.model.base_learning_schedule
        print(f"Setting learning rate to {model.learning_rate:.2e}")
        print(f"Setting learning optimizer to {model.learning_optimizer}")
        print(f"Setting learning schedule to {model.learning_schedule}")


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


        #### init trainer
        # init trainer_config specificly device
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        
        
        trainer_setup = TrainerSetup(config, lightning_config, trainer_config, opt, now, logdir, cfgdir, ckptdir, model)
        trainer_config, trainer_kwargs = trainer_setup.trainer_config, trainer_setup.trainer_kwargs
        
        trainer = Trainer(**trainer_config, logger=trainer_kwargs["logger"], callbacks=trainer_kwargs["callbacks"])

        # Step 4: 运行训练
        try:
            trainer.fit(model=model, datamodule=data)
        except Exception as e:
            print(f"Training error: {e}")

    except Exception as e:
        print(e)
        print("Failed to parse config files. Please check your syntax.")
