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

class InvResidualS1(nn.Module):
    def __init__(self, in_channels, out_channels, t):
        super().__init__()
        self.pointwise1 = nn.Sequential(nn.Conv1d(in_channels, out_channels=in_channels*t, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(in_channels*t),
                                            nn.ReLU6(inplace=True),
                                            )
        
        self.depthwise1 = nn.Sequential(nn.Conv2d(in_channels*t, in_channels*t, kernel_size=3, stride=1, padding=1, groups=in_channels*t, bias=False),
                                       nn.BatchNorm2d(in_channels),
                                            nn.ReLU6(inplace=True),
                                            )

        self.pointwise2 = nn.Sequential(nn.Conv1d(in_channels*t, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(in_channels*t),
                                            nn.ReLU6(inplace=True),
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
        self.pointwise1 = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=in_channels*t, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(in_channels*t),
                                            nn.ReLU6(inplace=True),
                                            )
        
        self.depthwise1 = nn.Sequential(nn.Conv2d(in_channels*t, in_channels*t, kernel_size=3, stride=1, padding=1, groups=in_channels*t, bias=False),
                                       nn.BatchNorm2d(in_channels),
                                            nn.ReLU6(inplace=True),
                                            )

        self.pointwise2 = nn.Sequential(nn.Conv1d(in_channels*t, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(in_channels*t),
                                            nn.ReLU6(inplace=True),
                                            )
        
    def forward(self, x):
        x = self.pointwise1(x)

        x = self.depthwise1(x)

        x = self.pointwise2(x)

        return x

class MobileNetv2_(nn.Module):
    def __init__(self, ):
        super().__init__()
        