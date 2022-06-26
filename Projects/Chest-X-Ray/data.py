from pytorch_lightning import LightningDataModule 
import os
import torch
from  torchvision.datasets import ImageFolder
import torchvision.io as io
import torchvision.transforms as T
from functools import partial
from torch.utils.data import DataLoader

class DataModule(LightningDataModule):
    def __init__(self,data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.image_read_func = partial(io.read_image,mode=io.image.ImageReadMode.RGB)
        self.batch_size = 64
        self.num_workers= 8
        self.train_transform = T.Compose([
                T.Resize(size = (256,256)),
                T.RandomRotation(degrees = (-20,+20)),
                T.CenterCrop(size=224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

        self.test_trasform = T.Compose([
                T.Resize(size = (224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])


    def setup(self, stage= None) -> None:
        self.train_data = ImageFolder(os.path.join(self.data_dir,'train'),transform=self.train_transform,)
        self.val_data = ImageFolder(os.path.join(self.data_dir,'val'),transform=self.test_trasform,)
        self.test_data = ImageFolder(os.path.join(self.data_dir,'test'),transform=self.test_trasform,)
        self.val_length = len(self.val_data)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)

    def val_dataloader(self) :
        return DataLoader(dataset=self.val_data,batch_size=self.batch_size,num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_data,batch_size=self.batch_size,num_workers=self.num_workers)