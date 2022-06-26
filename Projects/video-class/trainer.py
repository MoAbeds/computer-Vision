import torch 
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra

import os
from pytorch_lightning import Trainer ,LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from model import VideoClassification
from data import KinData



@hydra.main(config_name='config')

def main(cfg:DictConfig):
    dm = KinData(cfg)
    dm.setup()
    model  = VideoClassification(cfg)
    cfg = cfg.trainer
    callback = ModelCheckpoint(filename="{epoch}-{val_acc}",
                               monitor='val_acc',
                               save_last=True,
                               mode='max')

    trainer = Trainer(gpus=cfg.gpus,
                      benchmark=True,
                      callbacks=[callback],
                      check_val_every_n_epoch=cfg.check_val_every_n_epoch,
                      max_epochs=cfg.epochs,
                      precision=cfg.precision,
                      gradient_clip_val=cfg.gradient_clip_value)
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
