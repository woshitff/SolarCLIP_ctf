from functools import partial
import importlib

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

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

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else 0
        self.use_worker_init_fn = use_worker_init_fn

        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
            # self.__train_dataloader = DataLoader(self.datasets["train"], batch_size=self.batch_size, 
            #               shuffle=True, num_workers=self.num_workers)
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
            # self.__val_dataloader = DataLoader(self.datasets["validation"], batch_size=self.batch_size, 
                        #   shuffle=False, num_workers=self.num_workers)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
                    instantiate_from_config(data_cfg)
                    
    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k])) 
            for k in self.dataset_configs
        )
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        dataLoader = DataLoader(self.datasets["train"], batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers)
        return DataLoader(self.datasets["train"], batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers)

    def _val_dataloader(self, shuffle=False):
        dataloader = DataLoader(self.datasets["validation"], batch_size=self.batch_size, 
                          shuffle=shuffle, num_workers=self.num_workers)
        self.__val_dataloader = dataloader
        return DataLoader(self.datasets["validation"], batch_size=self.batch_size, 
                          shuffle=shuffle, num_workers=self.num_workers)

    def _test_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size, 
                          shuffle=shuffle, num_workers=self.num_workers)
    
    def _predict_dataloader(self):
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers)