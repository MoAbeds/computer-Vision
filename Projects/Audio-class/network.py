import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torchmetrics

class AudioNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
    self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
    self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
    self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
    self.flatten = nn.Flatten()
    self.linear = nn.Linear(128 * 5 * 4, 10)
    self.softmax = nn.Softmax(dim=1)

  def forward(self,x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.flatten(x)
    x = self.linear(x)
    x = self.softmax(x)
    return x

class LightModel(LightningModule):
    def __init__(self,):
        super().__init__()
        self.base = AudioNetwork()
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()
    def forward(self,x):
        x = self.base(x)
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