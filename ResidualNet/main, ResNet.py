import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Convlayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.bnorm = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        return self.bnorm(self.conv(x))

class shortcut(nn.Module):
    def __init__(self, x):
        super().__init__()
        y = None

        return y+x

    def forward(self,):

        pass

class ResNet_(nn.Module):
    def __init__(self,):
        super(self, ResNet_).__init__()

        pass

    def forward(self,):

        pass

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