from  pytorch_lightning import LightningModule
import torch
import torchmetrics
import torch.nn as nn
from math import sqrt
import torch.optim as optim
from enum import Enum
import torch.nn.functional as F
import torchmetrics
import torchvision.models.resnet as RES




class LightModel(LightningModule):
    def __init__(self,val_length):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        self.model.fc = nn.Linear(2048,2)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()
        self.val_length = val_length
    def forward(self,x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        #print(batch)
        x, y = batch
        y_hat = self(x)
        #loss = nn.CrossEntropyLoss(y_hat, y)
        loss = self.loss(y_hat, y)
        self.accuracy(y_hat, y)
        self.log('train_acc_step', self.accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc = (y_hat.argmax(-1) == y).sum().item()
        return acc

    def validation_epoch_end(self, validation_step_outputs):
        acc = 0
        for pred in validation_step_outputs:
            acc += pred
        acc = acc / self.val_length
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True) 

    def configure_optimizers(self):
        optim= torch.optim.Adam(self.parameters(), lr=0.02)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.8)
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }