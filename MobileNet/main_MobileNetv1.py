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
        self.reludepthwise = nn.ReLU6(inplace=True)

        self.pointwise = nn.Conv2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = 1,
        stride = 1,
        padding = 0,
        groups = 1
    )
        
        self.bnpointwise = nn.BatchNorm2d(out_channels)
        self.relupointwise = nn.ReLU6(inplace=True)

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
    def __init__(self, num_classes=1000):
        super(MobileNetv1_, self).__init__()
        self.conv1 = ConvBlock(
        in_channel=3,
        out_channel=32,
        stride=2)

        self.config = [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (1024, 2),
            (1024, 1),
        ]

        layers = []
        in_channels = 32
        for out_channels, stride in self.config:
            layers.append(DepthwiseSeparableConv(in_channels, out_channels, stride))

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

transform = transforms.Compose([transforms.Resize((256,256)), 
                                transforms.RandomHorizontalFlip(), 
                                transforms.ToTensor(), 
                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                    (0.2470, 0.2435, 0.2616)
                                                    )])

train_dataset = datasets.CIFAR10(root='./path', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./path', train=False, transform=transform, download=True)

test_dataloder = DataLoader(dataset=test_dataset, batch_size=256, shuffle=True)
train_dataloder = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = MobileNetv1_().to(device)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

optimizer = optim.Adam(model.parameters(), lr=0.0001,)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1):
    
    correct = 0
    total = 0
    total_loss = 0
    model.train()

    for images, lables in train_dataloder:
        images, lables = images.to(device), lables.to(device)

        output = model(images)

        loss = loss_fn(output, lables)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        y_hat = torch.argmax(output)
        y_hat = torch.softmax(y_hat)

acc = (output == lables).sum().item()
print(f"Epoch {epoch+1}, Loss={total_loss/len(train_dataloder):.4f}, Acc={acc:.4f}")

model.eval()
with torch.no_grad():
    corrects = 0
    totals = 0
    total_loss = 0

    output = model(images)

    loss = loss_fn(output, lables)

    total_loss += loss.item()

    y_hat = torch.argmax(output)
    y_hat = torch.softmax(y_hat)
    correct += (y_hat == lables).sum().item()
    total += lables.sum().item()

print("Test Accuracy:", corrects / totals)