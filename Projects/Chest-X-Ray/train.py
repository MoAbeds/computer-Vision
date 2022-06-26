import torch 
import torch.nn as nn
import torch.nn.functional as F

import os
from pytorch_lightning import Trainer ,LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import LightModel
from data import DataModule


logger = TensorBoardLogger("tb_logs", name="my_model")
data_dir = '/content/data/chest_xray'


def main():
    
    dm = DataModule(data_dir)
    dm.setup()
    model = LightModel(val_length=dm.val_length)
    callback = ModelCheckpoint(filename="{epoch}-{val_acc}",
                               monitor='val_acc',
                               save_last=True,
                               mode='max')


    trainer = Trainer(gpus=1,
                      benchmark=True,
                      max_epochs=10,
                      precision=16,
                      callbacks=[callback],
                      check_val_every_n_epoch=2 , 
                     gradient_clip_val=8,
                     logger=logger)
    trainer.fit(model, datamodule=dm)



if __name__ == "__main__":
    main() 