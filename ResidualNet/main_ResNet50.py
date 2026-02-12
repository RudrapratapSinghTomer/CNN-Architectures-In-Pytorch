import torch
import torch.nn as nn
from typing import Type
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, downsample, stride, expansion: int = 4):
        super(BottleNeck, self).__init__()
        self.expansion = expansion
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bnorm2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)

        out_channel = in_channel * expansion

        self.conv3 = nn.Conv2d(in_channel, out_channel*self.expansion, kernel_size=1, stride=stride, padding=1)
        self.bnorm3 = nn.BatchNorm2d(out_channel)
        self.relu3 = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x

        x = self.relu1(self.bnorm1(self.conv1(x)))
        x = self.relu2(self.bnorm2(self.conv2(x)))
        x = self.relu3(self.bnorm3(self.conv2(x)))

        if self.downsample is not None:
            identity = self.downsample(x)

        x += identity
        x = nn.ReLU(x)

        return x

class ResNet_50(nn.Module):
    def __init__(self, img_channel, num_layer = 50, num_channel = 1000):
        super(self, ResNet_50).__init__()
        if num_layer==50:
            layers = [3, 4, 6, 3]
            self.expansion = 4
        
        self.in_channnel = 64

        self.conv0 = nn.Conv2d(in_channels=img_channel, out_channels=self.in_channnel, kernel_size=7, stride=3, padding=2)
        self.bnorm0 = nn.BatchNorm2d(self.in_channnel)
        self.relu0 = nn.ReLU(inplace=True)
        self.maxpool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = self.makelayer(BottleNeck, 64, layers[0])
        self.conv2 = self.makelayer(BottleNeck, 128, layers[1], stride=2)
        self.conv3 = self.makelayer(BottleNeck, 128, layers[2], stride=2)
        self.conv4 = self.makelayer(BottleNeck, 128, layers[3], stride=2)

        self.avgpool0 = nn.AdaptiveAvgPool2d((1,1))
        self.fc0 = nn.Linear(512*self.expansion, num_channel)

    def makelayer(self, block: type[BottleNeck], out_channels, blocks, stride=2):
        if stride != 1:
            self.downsample = nn.Sequential(nn.Conv1d(in_channels=self.in_channnel, out_channels=out_channels, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(out_channels*self.expansion),
                                            )
        
        layers = []
        layers.append(BottleNeck(in_channel=self.in_channnel ,out_channel=out_channels, stride=stride, expansion=self.expansion, downsample=self.downsample))

        self.in_channnel = out_channels*self.expansion

        for i in range(1, blocks):
            layers.append(BottleNeck(self.in_channnel, out_channel=out_channels, expansion=self.expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool0(self.relu0(self.bnorm0(self.conv0(x))))

        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))

        x = self.avgpool0(x)

        x = nn.Flatten(x, 1)

        x = self.fc0(x)

        return x

transform = transforms.Compose([transforms.Resize((256,256)), 
                                    transforms.RandomHorizontalFlip(), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.5,) (0.5,))])

train_dataset = datasets.CIFAR10(root='./path', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./path', train=False, transform=transform, download=True)

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available else 'cup')

model = ResNet_50().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range():
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    for images, labels in train_dataloader:
        images, lables = images.to(device), labels.to(device)

        optimizer.zero_grad

        output = model(images)

        loss = loss_fn(output, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        total += len(lables)

        y_pred = torch.softmax(output)
        y_pred = torch.argmax(y_pred)

        correct += (y_pred == labels).sum().item()