from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from pytorch_lightning import LightningDataModule 

import os
import torch
from  torchvision.datasets import ImageFolder
import torchvision.io as io
import torchvision.transforms as T
from functools import partial
from torch.utils.data import DataLoader

class DataModule(LightningDataModule):
    def __init__(self,cfg:DictConfig) -> None:
        super().__init__()
        cfg.data.root = to_absolute_path(cfg.data.root)
        cfg.data.root = os.path.abspath(cfg.data.root)

        self.cfg = cfg.data
        self.dataset_path  = os.path.join(self.cfg.root,self.cfg.name)
        vaild_names = {"imagenette2", "imagewoof2"}
        if self.cfg.name not in vaild_names:
            raise ValueError(f'Incoorect {self.cfg.name} , vaild options {vaild_names}')

        if cfg.apply_resizer_model:
            img_size = self.cfg.resizer_image_size
        else:
            img_size = self.cfg.image_size

        self.image_read_func = partial(io.read_image,mode=io.image.ImageReadMode.RGB)

        self.train_transform = T.Compose([
            T.Resize((img_size,img_size)),
            T.RandomHorizontalFlip(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.test_trasform = T.Compose([
            T.Resize((img_size,img_size)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


    def setup(self, stage= None) -> None:
        self.train_data = ImageFolder(os.path.join(self.dataset_path,'train'),transform=self.train_transform,loader=self.image_read_func)
        self.val_data = ImageFolder(os.path.join(self.dataset_path,'val'),transform=self.train_transform,loader=self.image_read_func)
        self.val_length = len(self.val_data)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data,batch_size=self.cfg.batch_size,shuffle=True,num_workers=self.cfg.num_workers)

    def val_dataloader(self) :
        return DataLoader(dataset=self.val_data,batch_size=self.cfg.batch_size,num_workers=self.cfg.num_workers)