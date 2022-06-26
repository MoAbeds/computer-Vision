from pytorch_lightning import LightningDataModule 
import pandas as pd

import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader ,Dataset
import torchaudio

class UrbanDataset(Dataset):
  def __init__(self,ann_file,root_dir,target_sam_rate,num_sample):
    self.ann_file = pd.read_csv(ann_file)
    self.root_dir = root_dir
    self.target_sam_rate = target_sam_rate
    self.num_sample = num_sample

  def __len__(self):
    return len(self.ann_file)


  def __getitem__(self,index):
        audio_sample_path , label = self._get_audio_sample_path(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample(signal,sr)
        signal = self._mix_down(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad(signal)
        signal = self._melSpect(signal)
        
        return signal, label

  def _get_audio_sample_path(self, index):
        fold = f"fold{self.ann_file.iloc[index, 5]}"
        path = os.path.join(self.root_dir, fold, self.ann_file.iloc[
            index, 0])
        label = self.ann_file.iloc[index, 6]
        return path , label

  def _resample(self,signal,sr):
    if sr != self.target_sam_rate:
      resampler=torchaudio.transforms.Resample(sr,self.target_sam_rate)
      signal  =resampler(signal)

    return signal

  def _mix_down(self,signal):
    if signal.shape[0] > 1:
      signal = torch.mean(signal,dim=0 , keepdim=True)
    return signal

  def _melSpect(self,signal):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=self.target_sam_rate,
        n_fft=1024,
        hop_length=512,
        n_mels = 64
    )
    return mel(signal)

  def _cut_if_necessary(self,signal):
    if signal.shape[1] > self.num_sample:
      signal = signal [:,:self.num_sample]
    return signal

  def _right_pad(self,signal):
    length_signal = signal.shape[1]
    if length_signal < self.num_sample:
      num_missing_sam = self.num_sample - length_signal
      last_dim_padd = (0,num_missing_sam)
      signal = torch.nn.functional.pad(signal,last_dim_padd)
    return signal


class AudioLight(LightningDataModule):
    def __init__(self,ann_file,root_dir,sample_rate,num_sample,batch_size):
        super().__init__()
        self.data = UrbanDataset(ann_file,root_dir,sample_rate,num_sample)
        self.batch_size = batch_size
    def train_dataloader(self):
        return DataLoader(dataset=self.data,batch_size=self.batch_size,shuffle=True) 

    
