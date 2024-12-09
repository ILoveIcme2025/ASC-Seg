
import torch
from torch import nn
from mamba_ssm import Mamba



class PMB(nn.Module):
    
    def __init__(self, in_channels,pool_scale=4):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.mamba=Mamba(
                d_model=in_channels//4, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
        )
        self.proj = nn.Linear(in_channels, in_channels)
        self.pro1=nn.Conv2d(in_channels, in_channels,1)
        self.dwconv=nn.Conv2d(in_channels,in_channels,7,padding=3,groups=in_channels)
        self.gelu=nn.GELU()
        self.pro2=nn.Conv2d(in_channels, in_channels,1)
        

    def forward(self,x):
        B,C,H,W=x.shape
        x=x.view(B,C,H*W).permute(0,2,1).contiguous()
        x_norm=self.norm(x)
        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + x1
        x_mamba2 = self.mamba(x2) + x2
        x_mamba3 = self.mamba(x3) + x3
        x_mamba4 = self.mamba(x4) + x4
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        x_stage=x_mamba+x
        x_stage_nrom=self.norm(x_stage)
        x_stage_nrom=x_stage_nrom.permute(0,2,1).contiguous().view(B,C,H,W)
        x_ffn=self.pro1(x_stage_nrom)
        x_ffn=self.dwconv(x_ffn)
        x_ffn=self.gelu(x_ffn)
        x_ffn=self.pro2(x_ffn)
        return x_ffn


import torchvision

class DCB_Func(nn.Module):

    def __init__(self, in_channels, kernel_size=3,padding=1,dilation=1):
        super().__init__()
        self.offset=nn.Sequential(
            nn.Conv2d(in_channels,out_channels=2 * kernel_size * kernel_size,kernel_size=kernel_size,padding=kernel_size//2,dilation=dilation),
            nn.BatchNorm2d(2 * kernel_size * kernel_size)
        )
        self.deform=torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=in_channels,
                                                        dilation=dilation,
                                                        bias=False)
        self.balance=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        offsets = self.offset(x)
        out = self.deform(x, offsets)
        out = self.balance(out)
        return out


class DCB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pro=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.deform=DCB_Func(in_channels, kernel_size=3,padding=1)
        self.act=nn.GELU()

    def forward(self, x):
        shorcut = x
        x = self.pro(x)
        x = self.deform(x)
        x=self.act(x)*shorcut+shorcut
        return x



class ASEBlock(nn.Module):
    def __init__(self,input_channels=3,out_channels=3,pool=True,pool_scale=4):
        super().__init__()
        if pool:
            self.pool=nn.MaxPool2d(2, stride=2)
        else:
            self.pool=None
        self.ipc=nn.Sequential(
            nn.Conv2d(input_channels,out_channels,kernel_size=7,padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.pm=PMB(out_channels,pool_scale=pool_scale)
        self.dc=DCB(out_channels)

    def forward(self, x):
        if self.pool!=None:
            x=self.pool(x)
        x_ipc=self.ipc(x)
        x_ma=self.pm(x_ipc)
        x_def=self.dc(x_ipc)
        x=x_ma+x_def
        return x

class ASE(nn.Module):
    def __init__(self,input_channels=[64,128,256,512,512]):
        super().__init__()
        self.eb1=ASEBlock(3,input_channels[0],pool=False)
        self.eb2=ASEBlock(input_channels[0],input_channels[1])
        self.eb3=ASEBlock(input_channels[1],input_channels[2])
        self.eb4=ASEBlock(input_channels[2],input_channels[3])

    def forward(self, x):
        x1=self.eb1(x)
        x2=self.eb2(x1)
        x3=self.eb3(x2)
        x4=self.eb4(x3)
        x_list=[x1,x2,x3,x4]
        return x_list