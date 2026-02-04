import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

class MainConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(MainConv2d).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()

    def forward_pass(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x

class MainInception(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3, n5x5, max3x3):
        super(MainInception).__init__()
        self.Ix1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=n1x1, kernel_size=1),
                                 nn.ReLU(True), 
                                 )

        self.Ix2 = nn.Sequential(MainConv2d(in_channels=in_channels, out_channels=n1x1, kernel_size=1),
                                 MainConv2d(in_channels=n1x1, out_channels=n3x3, kernel_size=3, padding=1),
                                 )
        
        self.Ix3 = nn.Sequential(MainConv2d(in_channels=in_channels, out_channels=n1x1, kernel_size=1),
                                 MainConv2d(in_channels=n3x3, out_channels=n5x5, kernel_size=5, padding=2),
                                 )
        
        self.Ix4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                 MainConv2d(in_channels=in_channels, out_channels=n1x1, kernel_size=1),)

    def forward_pass(self, x):
        self.Iy1 = self.Ix1(x)
        self.Iy2 = self.Ix2(x)
        self.Iy3 = self.Ix3(x)
        self.Iy4 = self.Ix4(x)
        
        return torch.cat(self.Iy1, self.Iy2, self.Iy3. self.Iy4)

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.7):
        super(AuxiliaryClassifier).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3, padding=2)
        self.conv1 = nn.Sequential(MainConv2d(in_channels=in_channels, out_channels=128, kernel_size = 1),
                                   nn.ReLU(True),)
        self.flatten_ = nn.Flatten()
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.dropout_ = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward_pass(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.flatten_(x)
        x = self.fc1(x)
        x = self.dropout_(x)
        x = self.fc2(x)
        
        return x

class GoogleNet_(nn.Module):
    def __init__():
        super(GoogleNet_).__init__()
        pass

    def forward_pass(self, x):
        pass

transform = transforms.Compose([transforms.Resize((224,224)), 
                                transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor(), 
                                        transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.CIFAR10(root='./path', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./path' , train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

device = torch.device(f'CUDA' if torch.cuda.is_available() else 'CPU')

model = GoogleNet_()
optimizer = optim.Adam(model.parameters, lr=0.0001)
loss_fn = nn.CrossEntropyLoss

for epoch in range(5):
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    for images, lables in train_loader:
        images, lables = images.to(device), lables.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, lables)

        loss.backward()
        optimizer.step()

        total_loss = loss.item()

        y_pred = torch.argmax(output)
        correct = (y_pred == lables).sum().item()
        total = len(lables)

model.eval()