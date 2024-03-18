import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

class LoggerCallback(Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch_loss = torch.stack(pl_module.training_step_outputs).mean() # recheck this, use reduce='sum' and divide by total number of samples
        pl_module.log('train loss', epoch_loss, on_step=False, on_epoch=True)
        pl_module.training_step_outputs.clear()
        for tag, metric in pl_module.train_metrics.items():
            pl_module.log(tag, metric, on_step=False, on_epoch=True)
        
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch_loss = torch.stack(pl_module.validation_step_outputs).mean()
        pl_module.log('val loss', epoch_loss,  on_step=False, on_epoch=True)
        pl_module.validation_step_outputs.clear()
        for tag, metric in pl_module.val_metrics.items():
            pl_module.log(tag, metric, on_step=False, on_epoch=True)