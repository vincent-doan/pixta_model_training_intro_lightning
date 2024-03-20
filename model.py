import lightning as L
from torch import nn, optim
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score

class ResNet50LightningModule(L.LightningModule):
    def __init__(self, hidden_size:int, output_size:int, transfer:bool, optimizer_config:dict, loss_fn):
        super(ResNet50LightningModule, self).__init__()

        # OPTIMIZER & SCHEDULER
        self.learning_rate = optimizer_config["learning_rate"]
        self.step_size = optimizer_config["step_size"]
        self.gamma = optimizer_config["gamma"]
        
        # LOSS FUNCTION
        self.loss_fn = loss_fn

        # MODEL
        self.pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.pretrained_model.fc = nn.Linear(self.pretrained_model.fc.in_features, hidden_size)
        self.model = nn.Sequential(
            self.pretrained_model,
            nn.ReLU(),
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, output_size),
        )
        if transfer == 1:
            self.freeze_pretrained_weights()

        # METRICS STORING
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        self.train_acc = MultilabelAccuracy(num_labels=output_size)
        self.train_prec = MultilabelPrecision(num_labels=output_size, average='weighted')
        self.train_rec = MultilabelRecall(num_labels=output_size, average='weighted')
        self.train_f1 = MultilabelF1Score(num_labels=output_size, average='weighted')
        self.train_metrics = {
            'train/acc': self.train_acc,
            'train/prec': self.train_prec,
            'train/rec': self.train_rec,
            'train/f1': self.train_f1
        }
        
        self.val_acc = MultilabelAccuracy(num_labels=output_size)
        self.val_prec = MultilabelPrecision(num_labels=output_size, average='weighted')
        self.val_rec = MultilabelRecall(num_labels=output_size, average='weighted')
        self.val_f1 = MultilabelF1Score(num_labels=output_size, average='weighted')
        self.val_metrics = {
            'val/acc': self.val_acc,
            'val/prec': self.val_prec,
            'val/rec': self.val_rec,
            'val/f1': self.val_f1
        }

    # CORE LIGHTNING FUNCTIONS
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
    
    def training_step(self, train_batch, batch_idx):
        inputs, targets = train_batch
        preds = self.forward(inputs)
        # METRICS
        for metric in self.train_metrics.values():
            metric.update(preds, targets)
        # LOSS
        loss = self.loss_fn(preds, targets)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, train_batch, batch_idx):
        inputs, targets = train_batch
        preds = self.forward(inputs)
        # METRICS
        for metric in self.val_metrics.values():
            metric.update(preds, targets)
        # LOSS
        loss = self.loss_fn(preds, targets)
        self.validation_step_outputs.append(loss)

    # NON-LIGHTNING FUNCTIONS
    def freeze_pretrained_weights(self):
        for name, param in self.pretrained_model.named_parameters():
            if 'fc' in name:
                continue
            param.requires_grad = False

    def unfreeze_pretrained_weights(self):
        for name, param in self.pretrained_model.named_parameters():
            if 'fc' in name:
                continue
            param.requires_grad = True

    def adjust_learning_rate(self, new_scheduler):
        self.scheduler = new_scheduler