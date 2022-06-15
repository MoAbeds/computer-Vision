import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import pytorchvideo.data
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)
class KinData(LightningDataModule):
    def __init__(self, cfg:DictConfig):
        super().__init__()
        cfg.data.root = to_absolute_path(cfg.data.root)
        cfg.data.root = os.path.abspath(cfg.data.root)
        self.cfg = cfg
        self.dataset_path  = os.path.join(self.cfg.root,self.cfg.name)
        self.Clip_Dur = cfg.data.CLIP_DURATION
        self.Batch_size = cfg.data.BATCH_SIZE
        self.num_workers = cfg.data.NUM_WORKERS

    
    def setup(self) :
        train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(8),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    RandomHorizontalFlip(p=0.5),
                  ]
                ),
              ),
            ]
        )

        self.train_dataset = pytorchvideo.data.Kinetics(
            data_path = os.path.join(self.cfg.data.root,'train'),
            clip_sampler= pytorchvideo.data.make_clip_sampler('random',self.Clip_Dur),decode_audio=False,
            transform=train_transform
        )

        self.val_dataset = pytorchvideo.data.Kinetics(
            data_path = os.path.join(self.cfg.data.root,'val'),
            clip_sampler= pytorchvideo.data.make_clip_sampler('uniform',self.Clip_Dur),decode_audio=False
        )

    def train_dataloader(self):
        

        return DataLoader(
            self.train_dataset,batch_size=self.Batch_size,num_workers=self.num_workers
        )

    def val_dataloader(self):
        

        return DataLoader(
            self.val_dataset,batch_size=self.Batch_size,num_workers=self.num_workers
        )

