import torch
from torch import nn

class WeightedBCELoss(nn.Module):
    def __init__(self, output_size:int, scale:int, reduction:str):
        super(WeightedBCELoss, self).__init__()
        self.output_size = output_size
        self.scale = scale
        self.reduction = reduction

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        device = inputs.device
        loss = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.ones((batch_size, self.output_size), device=device)*self.scale)(inputs, targets)
        if self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)