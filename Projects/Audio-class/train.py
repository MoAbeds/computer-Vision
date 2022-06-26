import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from data import AudioLight
from network import LightModel
from pytorch_lightning.loggers import TensorBoardLogger


ann_file = '/content/UrbanSound8K/metadata/UrbanSound8K.csv'
Root_dir = '/content/UrbanSound8K/audio'
Sample_rate = 22050
num_sample = 22050
BATCH_SIZE = 128
logger = TensorBoardLogger("tb_logs", name="my_model")

def main():
    
    dm = AudioLight(ann_file,Root_dir,Sample_rate,num_sample,BATCH_SIZE)
    model = LightModel()



    trainer = Trainer(gpus=1,
                      benchmark=True,
                      max_epochs=10,
                      precision=16,
                     gradient_clip_val=8,logger=logger)
    trainer.fit(model, datamodule=dm)



if __name__ == "__main__":
    main() 