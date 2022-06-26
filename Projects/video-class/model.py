from typing import Dict
from  pytorch_lightning import LightningModule
import torch
import torchmetrics
import torch.nn as nn
from omegaconf import DictConfig
from math import sqrt
import torch.optim as optim
from enum import Enum
from pytorchvideo.models import resnet
import torch.nn.functional as F
import torchmetrics

def Resnet_Model(cfg:DictConfig):
    model = resnet.create_resnet(
        input_channel=cfg.network.input,
        model_depth=cfg.network.depth,
        model_num_class=cfg.network.classes,
        norm=nn.BatchNorm3d,
        activation=nn.ReLU

    )


class VideoClassification(LightningModule):
    def __init__(self,cfg:DictConfig):
        super().__init__()
        self.model = Resnet_Model(cfg)
        self.accuracy = torchmetrics.Accuracy()
        self.cfg = cfg
    def forward(self,x):
        return self.model(x)

    def training_step(self,batch,batch_idx):
        y_hat = self.model(batch['video'])
        loss = F.cross_entropy(y_hat,batch['label'])
        self.accuracy(y_hat, batch['label'])
        self.log('train_loss,',loss.item(),self.accuracy)
        return loss

    def validation_step(self,batch,batch_idx ):
        y_hat = self.model(batch['video'])
        loss = F.cross_entropy(y_hat,batch['label'])
        self.log('val_loss',loss)
        return loss

    def configure_optimizers(self):
      optimizer = optim.Adam(self.parameters(),lr =self.cfg.network.lr )       



