import matplotlib.pyplot as plt
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger

class LoggerCallback(Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # LOSS
        dataset_length = len(trainer.train_dataloader.dataset)
        epoch_loss = sum([loss.item() for loss in pl_module.training_step_outputs])/dataset_length
        pl_module.log('train/loss', epoch_loss, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.training_step_outputs.clear()
        # METRICS
        for tag, metric in pl_module.train_metrics.items():
            pl_module.log(tag, metric, on_step=False, on_epoch=True, sync_dist=True)
        
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # LOSS
        dataset_length = len(trainer.val_dataloaders.dataset)
        epoch_loss = sum([loss.item() for loss in pl_module.validation_step_outputs])/dataset_length
        pl_module.log('val/loss', epoch_loss,  on_step=False, on_epoch=True, sync_dist=True)
        pl_module.validation_step_outputs.clear()
        # METRICS
        for tag, metric in pl_module.val_metrics.items():
            pl_module.log(tag, metric, on_step=False, on_epoch=True, sync_dist=True)
    
class VisualizeFilterCallback(Callback):
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        first_conv_layer = pl_module.pretrained_model.conv1
        num_filters = first_conv_layer.weight.size(0)
        first_conv_layer_weights = first_conv_layer.weight.data
        
        weights_min, weights_max = first_conv_layer_weights.min(), first_conv_layer_weights.max()
        normalized_weights = (first_conv_layer_weights - weights_min) / (weights_max - weights_min)
        fig, axs = plt.subplots(num_filters // 8, 8, figsize=(12, 12))

        for i in range(num_filters // 8):
            for j in range(8):
                axs[i, j].imshow(normalized_weights[i * 8 + j, 0].cpu(), cmap='viridis')
                axs[i, j].axis('off')

        filter_visualization_path = 'filters/filters_visualization_{0}.png'.format(trainer.current_epoch)
        plt.savefig(filter_visualization_path, format='png')
        
        logger = trainer.logger
        if isinstance(logger, TensorBoardLogger):
            img = plt.imread(filter_visualization_path)
            logger.experiment.add_image('filters/first_layer', img, global_step=trainer.current_epoch, dataformats='HWC')