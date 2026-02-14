import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
        in_channels= in_channels,
        out_channels= in_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups= in_channels
    )
        self.bndepthwise = nn.BatchNorm2d(in_channels)
        self.reludepthwise = nn.ReLU(inplace=True)

        self.pointwise = nn.Conv2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = 1,
        stride = 1,
        padding = 0,
        groups = 1
    )
        
        self.bnpointwise = nn.BatchNorm2d(out_channels)
        self.relupointwise = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bndepthwise(x)
        x = self.reludepthwise(x)
        
        x = self.pointwise(x)
        x = self.bnpointwise(x)
        x = self.relupointwise(x)
        
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channel)
        self.relu0 = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.relu0(self.bn0(self.conv0(x)))

        return x
    
class MobileNetv1_(nn.Module):
    def __init__(self,):
        super(MobileNetv1_, self).__init__()
        self.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=32,
        kernel_size=3,
        stride=2,
        padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,):
        
        pass

transform = transforms.Compose([transforms.Resize((256,256)), 
                                transforms.RandomHorizontalFlip(), 
                                transforms.ToTensor(), 
                                transforms.Normalize()])

test_dataset = datasets.CIFAR10(root='./path', train=True, transform=transform, download=False)
train_dataset = datasets.CIFAR10(root='./path', train=False, transform=transform, download=False)

test_dataloder = DataLoader(dataset=test_dataset, batch_size=256, shuffle=True)
test_dataloder = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MobileNetv1_().to(device)

model.train()
optimizer = optim.Adam(model.parameters, lr=0.0001,)
loss_fn = nn.CrossEntropyLoss()