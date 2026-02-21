import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, first=False):
        super().__init__()
        if first:
            in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False

        else:
            in_channels=320, out_channels=1280, kernel_size=1, stride=1, padding=0, bias=False

        self.conv0 = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                out_channels=out_channels, 
                                                    kernel_size=kernel_size, 
                                                        stride=stride, 
                                                            padding=padding),
                                   nn.BatchNorm2d(num_features=out_channels),
                                   nn.ReLU6(inplace=True),
                                   )
        
    def forward(self, x):
        x = self.conv0(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t):
        super(self, InvertedResidual).__init__()
        x
        pass

    def forward(self, x):

        pass

class InvResidualS1(nn.Module):
    def __init__(self, in_channels, out_channels, t):
        super().__init__()
        self.pointwise1 = nn.Sequential(nn.Conv2d(in_channels, out_channels=in_channels*t, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(in_channels*t),
                                            nn.ReLU6(inplace=True),
                                            )
        
        self.depthwise1 = nn.Sequential(nn.Conv2d(in_channels*t, in_channels*t, kernel_size=3, stride=1, padding=1, groups=in_channels*t, bias=False),
                                       nn.BatchNorm2d(in_channels*t),
                                            nn.ReLU6(inplace=True),
                                            )

        self.pointwise2 = nn.Sequential(nn.Conv2d(in_channels*t, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(in_channels*t),
                                            )

    def forward(self, x):
        identity = x

        x = self.pointwise1(x)

        x = self.depthwise1(x)

        x = self.pointwise2(x)

        x += identity

        return x
    
class InvResidualS2(nn.Module):
    def __init__(self, in_channels, out_channels, t):
        super().__init__()
        self.pointwise1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels*t, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(in_channels*t),
                                            nn.ReLU6(inplace=True),
                                            )
        
        self.depthwise1 = nn.Sequential(nn.Conv2d(in_channels*t, in_channels*t, kernel_size=3, stride=2, padding=1, groups=in_channels*t, bias=False),
                                       nn.BatchNorm2d(in_channels*t),
                                            nn.ReLU6(inplace=True),
                                            )

        self.pointwise2 = nn.Sequential(nn.Conv2d(in_channels*t, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(in_channels*t),
                                            )
        
    def forward(self, x):
        x = self.pointwise1(x)

        x = self.depthwise1(x)

        x = self.pointwise2(x)

        return x

class MobileNetv2_(nn.Module):
    '''
    Docstring for MobileNetv2_
    Here t=expansion, c=out_channel, n=num_blocks, s=stride
    '''
    def __init__(self, ):
        super().__init__()
        #224^2 * 3 Conv2d t=-, c=32, n=1, s=2
        self.conv0 = ConvBlock(first=True)

        config = [# t, c, n, s
                        (1, 16, 1, 1),
                        (6, 24, 2, 2),
                        (6, 32, 3, 2),
                        (6, 64, 4, 2),
                        (6, 96, 3, 1),
                        (6, 160, 3, 2),
                        (6, 320, 1, 1),]

        #112^2 * 32 bottleneck t=1, c=16, n=1, s=1
        self.conv1 = InvResidualS1(in_channels=32, out_channels=16, t=1)
        
        #112^2 * 16 bottleneck t=6, c=24, n=2, s=2
        self.in_channels=16
        self.features = nn.ModuleList()
        for t,c,n,s in config:
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(inv)

        self.conv2 = nn.ModuleList
        pass

    def forward(self, x):

        pass