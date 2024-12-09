
import torch
import torch.nn as nn

class MSFABlock(nn.Module):
    def __init__(self,in_channels,padding,dilation,sample1=None,sample2=None):
        super().__init__()
        self.sample1=sample1
        self.sample2=sample2
        self.extract=nn.Sequential(
            nn.Conv2d(in_channels,in_channels//2,kernel_size=3,padding=padding,dilation=dilation),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU()
        )
        self.integrate=nn.Sequential(
            nn.Conv2d(in_channels//2,in_channels//2,1),
            nn.BatchNorm2d(in_channels//2)
        )

    def forward(self,x):
        if self.sample1!=None:
            x=self.sample1(x)
        x=self.extract(x)
        x=self.integrate(x)
        if self.sample2!=None:
            x=self.sample2(x)
        return x



class CAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.score1=nn.AdaptiveMaxPool2d(1)
        self.score2=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        score=self.score1(x) + self.score2(x)
        score=self.conv(score)
        x=score*x+x
        return x



class SAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ave=nn.Sequential(
            nn.AvgPool2d(3,1,1),
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels,1,1),
            nn.BatchNorm2d(1)
        )
        self.max=nn.Sequential(
            nn.MaxPool2d(3,1,1),
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels,1,1),
            nn.BatchNorm2d(1)
        )
        self.conv=nn.Sequential(
            nn.Conv2d(2,1,7,padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_ave=self.ave(x)
        x_max=self.max(x)
        x_=torch.cat([x_ave,x_max],dim=1)
        x=self.conv(x_)*x+x
        return x



class DFB(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.AP1=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,7,padding=3,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
        )
        self.AP2=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,5,padding=2,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
        )
        self.AP3=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,padding=1,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
        )
        self.AP=nn.AvgPool2d(3,1,1)
        self.sig=nn.Sigmoid()
        self.weight=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )
        self.norm1=nn.BatchNorm2d(in_channels)
        self.norm2=nn.BatchNorm2d(in_channels)

    def forward(self,x):
        x_norm=self.norm1(x)
        x1=self.AP1(x_norm)
        x2=self.AP2(x_norm)
        x3=self.AP3(x_norm)
        x_sum=x1+x2+x3
        x_sum=self.norm2(x_sum)
        x_ap=x_sum-self.AP(x_sum)
        x_sig=self.weight(x_ap)
        x_ee=x_sum*x_sig+x_sum
        return x_ee



class SFM(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.attention=nn.Sequential(
            DFB(in_channels),
            SAB(in_channels),
            CAB(in_channels)
        )

    def forward(self,x):
        x=self.attention(x)
        return x




class MSFA(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.cfe1=MSFABlock(in_channels,1,1)
        self.cfe2=MSFABlock(in_channels,2,2)
        self.cfe3=MSFABlock(in_channels,3,3)
        self.cfe4=MSFABlock(in_channels,4,4)
        self.cfe5=MSFABlock(in_channels,1,1,nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),nn.MaxPool2d(kernel_size=2,stride=2))
        self.cfe6=MSFABlock(in_channels,3,3,nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),nn.MaxPool2d(kernel_size=2,stride=2))
        self.cfe7=MSFABlock(in_channels,1,1,nn.MaxPool2d(kernel_size=2,stride=2),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.cfe8=MSFABlock(in_channels,3,3,nn.MaxPool2d(kernel_size=2,stride=2),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.extract=nn.Sequential(
            nn.Conv2d(4*in_channels,in_channels,3,padding=1,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels)
        )
        
    def forward(self,x):
        x1=self.cfe1(x)
        x2=self.cfe2(x)
        x3=self.cfe3(x)
        x4=self.cfe4(x)
        x5=self.cfe5(x)
        x6=self.cfe6(x)
        x7=self.cfe7(x)
        x8=self.cfe8(x)
        out=torch.cat([x1,x2,x3,x4,x5,x6,x7,x8],dim=1)
        out=self.extract(out)
        return out




class CNSD(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.msfa=MSFA(in_channels)
        self.sf=SFM(in_channels)
        self.relu=nn.ReLU()

    def forward(self,x):
        short_cut=x
        x=self.msfa(x)
        x=self.relu(self.sf(x)+short_cut)
        return x
    
class FAC(nn.Module):
    def __init__(self,in_channels1,in_channels2):
        super().__init__()
        self.pro=nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels1,in_channels2,1),
            nn.BatchNorm2d(in_channels2),
            nn.Sigmoid()
        )
        self.sig=nn.Sigmoid()

    def forward(self,x1,x2):
        x=self.pro(x1)*self.sig(x2)+x2
        return x



class Decoder(nn.Module):
    def __init__(self,in_channels=[64,128,256,512,512]):
        super().__init__()
        self.num_layer=len(in_channels)
        self.cnsd=nn.ModuleList()
        for i_layer in range(self.num_layer):
            self.cnsd.append(CNSD(in_channels[i_layer]))
        self.fac=nn.ModuleList()
        for i_layer in range(self.num_layer-1):
            self.fac.append(FAC(in_channels[i_layer+1],in_channels[i_layer]))
        
    def forward(self,x):
        x_list=[]
        input=x[-1]
        for i in range(-1, -len(self.cnsd)-1, -1):
            x_d=self.cnsd[i](input)
            x_list.append(x_d)
            if i!=-self.num_layer:
                input=self.fac[i](x_d,x[i-1])
        return x_list



class PH_Block(nn.Module):
    def __init__(self,in_channels,scale_factor=1):
        super().__init__()
        if scale_factor>1:
            self.upsample=nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.upsample=None
        self.pro=nn.Sequential(
            nn.Conv2d(in_channels,1,1),
            nn.Sigmoid()
        )

    def forward(self,x:torch.Tensor):
        if self.upsample!=None:
            x=self.upsample(x)
        x=self.pro(x)
        return x



class PredictionHead(nn.Module):
    def __init__(self,in_channels=[32,64,96,128,160],scale_factor=[1,2,4,8,16]):
        super().__init__()
        self.ph1=PH_Block(in_channels[0],scale_factor[0])
        self.ph2=PH_Block(in_channels[1],scale_factor[1])
        self.ph3=PH_Block(in_channels[2],scale_factor[2])
        self.ph4=PH_Block(in_channels[3],scale_factor[3])

    def forward(self,x):
        x4,x3,x2,x1=x

        x1=self.ph1(x1)
        x2=self.ph2(x2)
        x3=self.ph3(x3)
        x4=self.ph4(x4)

        x_list=[x1,x2,x3,x4]
        return x_list