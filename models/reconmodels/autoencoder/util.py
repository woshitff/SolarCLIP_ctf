import importlib

import torch

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def config_optimizers(optimizer_name, parameters, learning_rate, lr_scheduler_name=None):
    # 选择优化器
    if optimizer_name == 'Adam':
        opt = torch.optim.Adam(parameters, lr=learning_rate)
    elif optimizer_name == 'SGD':
        opt = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"optimizer {optimizer_name} is not supported")

    # 选择学习率调度器
    if lr_scheduler_name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, verbose=True)
    elif lr_scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    elif lr_scheduler_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000, eta_min=0)
    else:
        scheduler = None
    
    return [opt], [scheduler] if scheduler else [opt]