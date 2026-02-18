import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channel, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, groups=in_channel),
                                       nn.BatchNorm2d(in_channel),
                                            nn.ReLU6(inplace=True),
                                            )
        
        self.pointwise = nn.Sequential(nn.Conv1d(in_channel, out_channels=out_channels, kernel_size=1, stride=stride, padding=1, groups=in_channel),
                                       nn.BatchNorm2d(in_channel),
                                            nn.ReLU6(inplace=True),
                                            )
        
    def forward(self, x):
        x = self.depthwise(x)

        x = self.pointwise(x)

        return x
    

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expancion = expansion

        self.conv0 = nn.Conv2d()

        if self.stride == 2:
            self.NetForStrideOne = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels*expansion, kernel_size=(1,1), bias=False), 
                                                DepthwiseSeparableConv(in_channels*expansion)
                                                )
            
            self.NetForStrideTwo = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels*expansion, kernel_size=(1,1), bias=False), 
                                                 DepthwiseSeparableConv(in_channels*expansion, stride=(2,2))
                                                )

    def forward(self, input_image):
        if self.expansion == 1:
            x = self.depth(input_image)
            x = self.pw(x)
        else:
            x = self.conv1(input_image)
            x = self.dw(x)
            x = self.pw(x)

        # If input channel and output channel are same, then perform add
        # residual part
        if self.in_fts == self.out_fts:
            x = input_image + x          

        return x
    

class MobileNetv2_(nn.Module):
    def __init__(self, ):
        super().__init__()
        