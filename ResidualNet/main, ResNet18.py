import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, expancation, downsample, stride):
        super(BasicBlock, self).__init__()
        self.expancation = expancation
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample is not None:
            self.downsample(x)

        x += identity
        x = nn.ReLU(x)

        return x

class ResNet_(nn.Module):
    def __init__(self, img_size, num_layers, num_channel):
        super(self, ResNet_).__init__()
        if num_channel==18:
            layers = [2, 2, 2, 2]
            self.expansion = 1

        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=img_size, out_channels=self.in_channels, kernel_size=7, stride=3, padding=2)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu1 = nn.ReLU(True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = self.makelayer(BasicBlock, 64, layers[0])
        self.conv3 = self.makelayer(BasicBlock, 128, layers[1], padding=2)
        self.conv4 = self.makelayer(BasicBlock, 256, layers[2], padding=2)
        self.conv5 = self.makelayer(BasicBlock, 512, layers[3], padding=2)

        self.avg_pool1 = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512*self.expansion, out_features=num_channel)
        pass

    def makelayer(self, out_channels, blocks, stride):
        if stride != 1:
            downsample = nn.Sequential(BasicBlock(in_channel=self.in_channels, out_channel=out_channels, kernal_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_channels*self.expansion),
                                       )
            
            layers = []
            layers.append(BasicBlock(in_channel=self.in_channels, out_channel=out_channels, stride=stride, downsample=downsample))

        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channel=out_channels, expancation=self.expansion))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.max_pool1(self.relu1(self.bn1(self.conv1(x))))

        x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))

        x = self.avg_pool1(x)

        x = nn.Flatten(x, 1)

        x = self.fc1(x)

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

model = ResNet_()

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

model.eval()