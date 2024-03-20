import argparse, os
from model import ResNet50LightningModule
from loss import WeightedBCELoss
from data import PascalVOCDataModule
from callback import LoggerCallback, VisualizeFilterCallback
from torchvision.models import ResNet50_Weights
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

def main():

    # COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='Input batch size', default=32)
    parser.add_argument('--hidden_size', help='Input hidden size', default=128)
    parser.add_argument('--output_size', help='Input output size', default=20)
    parser.add_argument('--transfer', help='Choose whether to freeze pretrained weights', default=0)
    parser.add_argument('--pos_scale', help='Input positive label scale', default=5)
    parser.add_argument('--learning_rate', help='Input learning rate', default=0.0001)
    parser.add_argument('--step_size', help='Input step size', default=10)
    parser.add_argument('--gamma', help='Input gamma', default=0.3)
    parser.add_argument('--num_epochs', help='Input number of epochs', default=10)
    parser.add_argument('--continue_training', help='Choose whether to continue from a checkpoint', default=0)
    parser.add_argument('--checkpoint', help='Choose whether to use best or latest checkpoint', default='best')
    args = parser.parse_args()
    
    # LOSS FUNCTION
    loss_fn = WeightedBCELoss(
        output_size=int(args.output_size),
        scale=int(args.pos_scale),
        reduction='sum'
    )

    # MODEL
    best_path, last_path = tuple(sorted(os.listdir('checkpoints/')))
    if int(args.continue_training) == 1:
        if args.checkpoint == 'best':
            model = ResNet50LightningModule.load_from_checkpoint('checkpoints/' + best_path)
        elif args.checkpoint == 'last':
            model = ResNet50LightningModule.load_from_checkpoint('checkpoints/' + last_path) 
    else:
        model = ResNet50LightningModule(
            hidden_size=int(args.hidden_size),
            output_size=int(args.output_size),
            transfer=int(args.transfer),
            optimizer_config={
                "learning_rate": float(args.learning_rate),
                "step_size": int(args.step_size),
                "gamma": float(args.gamma)
            },
            loss_fn=loss_fn
        )

    # DATA
    data = PascalVOCDataModule(
        train_images_path='pascal_voc_2007_data/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/',
        val_images_path='pascal_voc_2007_data/voctest_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/',
        train_labels_path='labels/processed_trainval_labels.csv',
        val_labels_path='labels/processed_test_labels.csv',
        preprocess=ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=False),
        batch_size=int(args.batch_size)
    )

    # CALLBACKS
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/', monitor='val/loss', save_last=True, mode='min', save_on_train_epoch_end=True)
    tqdm_callback = TQDMProgressBar()
    lr_monitor = LearningRateMonitor()
    logger_callback = LoggerCallback()
    # visualize_filter_callback = VisualizeFilterCallback()

    callbacks = [
        logger_callback,
        # visualize_filter_callback,
        lr_monitor,
        checkpoint_callback,
        tqdm_callback,
    ]

    # LOGGER
    logger = TensorBoardLogger(save_dir='runs/')

    # TRAINER
    trainer = Trainer(callbacks=callbacks,
                      logger=logger,
                      min_epochs=int(args.num_epochs),
                      max_epochs=int(args.num_epochs) + 10,
                      log_every_n_steps=1
    )
    trainer.fit(model=model, datamodule=data)

if __name__ == '__main__':
    seed_everything(42, workers=True)
    main()