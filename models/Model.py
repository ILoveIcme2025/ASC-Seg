import torch
import torch.nn as nn

from models.Decoder import Decoder,PredictionHead
from models.Encoder import ASE

class Model(nn.Module):
    def __init__(self,input_channels=[64,128,256,512,512],scale_factor=[]):
        super().__init__()
        self.encoder=ASE(input_channels)
        self.decoder=Decoder(input_channels)
        self.final=PredictionHead(input_channels,scale_factor)

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        x=self.final(x)
        return x
 