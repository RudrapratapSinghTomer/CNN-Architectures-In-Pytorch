import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ConvBlock(nn.Module):
    def __init__(self, first):
        super().__init__()
        if first:
            self.conv0 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
                                                    nn.BatchNorm2d(32),
                                                    nn.ReLU6(inplace=True),
                                                    )

        else:
            self.conv0 = nn.Sequential(nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=1, stride=1, padding=0, bias=False),
                                                    nn.BatchNorm2d(1280),
                                                    nn.ReLU6(inplace=True),
                                                    )
        
    def forward(self, x):
        x = self.conv0(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t):
        super().__init__()

        hidden_dim = in_channels * t
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []
        #PointWise
        if t !=1:
            layers.append(nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(hidden_dim),
                                        nn.ReLU6(inplace=True),
                                        ))

        #DepthWise
        layers.append(nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(out_channels=hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
   
        #PointWise
        layers.append(nn.Sequential(nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        ))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileNetv2_(nn.Module):
    '''
    Docstring for MobileNetv2_
    Here t=expansion, c=out_channel, n=num_blocks, s=stride
    '''
    def __init__(self, num_classes=10):
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
        
        #112^2 * 16 bottleneck t=6, c=24, n=2, s=2
        self.features = nn.ModuleList()
        self.in_channels=32

        for t,c,n,s in config:
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(in_channels=self.in_channels, out_channels=c, stride=stride, t=t))

                self.in_channels = c
        
        self.conv1 = ConvBlock(first=False)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.conv0(x)

        for layer in self.features:
            x = layer(x)

        x = self.conv1(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc(x)

        return x