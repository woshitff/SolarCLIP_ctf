from typing import Optional

from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class LinearBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x= self.fc(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.ln(x)
        return x
        
class Predictor(nn.Module):
    def __init__(self, num_classes: int=6, hidden_size: int=768, hidden_layer: int=2, dropout = 0.1, **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.layers = nn.ModuleList([LinearBlock(hidden_size,dropout) for _ in range(hidden_layer)])
        # TODO not sure how to handel seq_len
        self.pooling = nn.AdaptiveAvgPool1d(1)
        # self.pooling = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: [batch_size, seq_len, hidden_size]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.pooling(hidden_states.permute(0, 2, 1)).squeeze(-1) # [batch_size, hidden_size]
        hidden_states = self.dropout(hidden_states)

        return self.head(hidden_states)
    
class SolarPredictor(pl.LightningModule):
    def __init__(self, model: SolarVivit, predictor: Predictor, only_train_predictor = False, criterion: str = 'CrossEntropyLoss', optimizer: str = 'adamw', lr: float = 1e-4, scheduler: Optional[OmegaConf] = None):
        super().__init__()
        self.model = model
        self.predictor = predictor

        self.only_train_predictor = only_train_predictor
        criterion_class = {'CrossEntropyLoss': nn.CrossEntropyLoss, 'MSELoss': nn.MSELoss}
        self.criterion = criterion_class[criterion](reduce = False)
        self.optimizer = optimizer
        self.lr = lr
        self.scheduler = scheduler
        if self.model.config.task == 'only_predict_next_frame':
            self.num_token_per_tubelet = self.model.num_token_per_tubelet
        elif self.model.config.task == 'next_tubelet':
            self.num_token_per_tubelet = self.model.num_token_per_tubelet

    # pytorch_lightning functions
    def forward(self, video_in, date_ids, LaLos, labels = None, focal_weight = None):
        if self.only_train_predictor:
            self.model.eval()
        hidden_states = self.model.encode(video_in=video_in, date_ids=date_ids, LaLos=LaLos)
        hidden_states = hidden_states[:, -self.num_token_per_tubelet:, :]
        logits = self.predictor(hidden_states)

        if labels is not None:
            original_loss = self.criterion(logits, labels)
            focal_loss = original_loss * focal_weight
            return logits, focal_loss.mean(), original_loss.mean()
        else:
            return logits, None, None
    
    def training_step(self, batch, batch_idx):
        video_in, date_ids, LaLos, labels, focal_weight = batch
        logits, focal_loss, original_loss = self(video_in, date_ids, LaLos, labels, focal_weight)
        acc = (logits.argmax(-1) == labels).float().mean()
        acc_0 = (logits.argmax(-1)[ labels == 0] == labels[labels == 0]).float().mean()
        acc_A = (logits.argmax(-1)[ labels == 1] == labels[labels == 1]).float().mean()
        acc_B = (logits.argmax(-1)[ labels == 2] == labels[labels == 2]).float().mean()
        acc_C = (logits.argmax(-1)[ labels == 3] == labels[labels == 3]).float().mean()
        acc_M = (logits.argmax(-1)[ labels == 4] == labels[labels == 4]).float().mean()
        acc_X = (logits.argmax(-1)[ labels == 5] == labels[labels == 5]).float().mean()
        acc_mean = []
        
        self.log('loss_train', focal_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss_train_original', original_loss, on_epoch=True, logger=True)
        self.log('acc_train', acc, on_epoch=True, logger=True)
        if not acc_0.isnan():
            self.log('acc_train_0', acc_0, on_epoch=True, logger=True)
            acc_mean.append(acc_0)
        if not acc_A.isnan():
            self.log('acc_train_A', acc_A, on_epoch=True, logger=True)
            acc_mean.append(acc_A)
        if not acc_B.isnan():
            self.log('acc_train_B', acc_B, on_epoch=True, logger=True)
            acc_mean.append(acc_B)
        if not acc_C.isnan():
            self.log('acc_train_C', acc_C, on_epoch=True, logger=True)
            acc_mean.append(acc_C)
        if not acc_M.isnan():
            self.log('acc_train_M', acc_M, on_epoch=True, logger=True)
            acc_mean.append(acc_M)
        if not acc_X.isnan():
            self.log('acc_train_X', acc_X, on_epoch=True, logger=True)
            acc_mean.append(acc_X)
        if len(acc_mean) > 0:
            acc_mean = torch.stack(acc_mean).mean()
        else:
            acc_mean = 0
        self.log('acc_train_mean', acc_mean, on_epoch=True, logger=True)

        return {'loss': focal_loss, 'acc': acc, 'acc_mean': acc_mean}
    
    def validation_step(self, batch, batch_idx):
        video_in, date_ids, LaLos, labels, focal_weight = batch
        logits, focal_loss, original_loss = self(video_in, date_ids, LaLos, labels, focal_weight)
        acc = (logits.argmax(-1) == labels).float().mean()
        acc_0 = (logits.argmax(-1)[ labels == 0] == labels[labels == 0]).float().mean()
        acc_A = (logits.argmax(-1)[ labels == 1] == labels[labels == 1]).float().mean()
        acc_B = (logits.argmax(-1)[ labels == 2] == labels[labels == 2]).float().mean()
        acc_C = (logits.argmax(-1)[ labels == 3] == labels[labels == 3]).float().mean()
        acc_M = (logits.argmax(-1)[ labels == 4] == labels[labels == 4]).float().mean()
        acc_X = (logits.argmax(-1)[ labels == 5] == labels[labels == 5]).float().mean()
        acc_mean = []
        
        self.log('loss_val', focal_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss_val_original', original_loss, on_epoch=True, logger=True)
        self.log('acc_val', acc, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if not acc_0.isnan():
            self.log('acc_val_0', acc_0, on_epoch=True, logger=True)
            acc_mean.append(acc_0)
        if not acc_A.isnan():
            self.log('acc_val_A', acc_A, on_epoch=True, logger=True)
            acc_mean.append(acc_A)
        if not acc_B.isnan():
            self.log('acc_val_B', acc_B, on_epoch=True, logger=True)
            acc_mean.append(acc_B)
        if not acc_C.isnan():
            self.log('acc_val_C', acc_C, on_epoch=True, logger=True)
            acc_mean.append(acc_C)
        if not acc_M.isnan():
            self.log('acc_val_M', acc_M, on_epoch=True, logger=True)
            acc_mean.append(acc_M)
        if not acc_X.isnan():
            self.log('acc_val_X', acc_X, on_epoch=True, logger=True)
            acc_mean.append(acc_X)
        if len(acc_mean) > 0:
            acc_mean = torch.stack(acc_mean).mean()
        else:
            acc_mean = 0
        self.log('acc_val_mean', acc_mean, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return {'loss': focal_loss, 'acc': acc, 'acc_mean': acc_mean}
    
    def configure_optimizers(self):
        optimizer_class = {'adamw': optim.AdamW, 'adam': optim.Adam, 'sgd': optim.SGD}
        if self.optimizer not in optimizer_class.keys():
            print(f'Optimizer {self.optimizer} not supported. Using AdamW instead.')
            optimizer_class = optim.AdamW
        else:
            optimizer_class = optimizer_class[self.optimizer]

        if self.only_train_predictor:
            optimizer = optimizer_class(self.predictor.parameters(), lr=self.lr, weight_decay= 0.1)
        else:
            optimizer = optimizer_class(self.parameters(), lr=self.lr, weight_decay= 0.1)

        if not self.scheduler:
            return optimizer
        else:
            scheduler_class = {'cosine': optim.lr_scheduler.CosineAnnealingLR, 'step': optim.lr_scheduler.StepLR, 'cosine_warmup': optim.lr_scheduler.CosineAnnealingWarmRestarts}
            if self.scheduler.type not in scheduler_class.keys():
                print(f'Scheduler {self.scheduler.type} not supported, do not use scheduler.')
                return optimizer
            else:
                scheduler_class = scheduler_class[self.scheduler.type]
                scheduler = scheduler_class(optimizer, **self.scheduler.params)
                return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': self.scheduler.interval}}
 
        