import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

class MainConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(MainConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x

class MainInception(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_, n3x3, n5x5_, n5x5, poolproj):
        super(MainInception, self).__init__()
        self.Ix1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=n1x1, kernel_size=1),
                                 nn.ReLU(True), 
                                 )

        self.Ix2 = nn.Sequential(MainConv2d(in_channels=in_channels, out_channels=n3x3_, kernel_size=1),
                                 MainConv2d(in_channels=n3x3_, out_channels=n3x3, kernel_size=3, padding=1),
                                 )
        
        self.Ix3 = nn.Sequential(MainConv2d(in_channels=in_channels, out_channels=n5x5_, kernel_size=1),
                                 MainConv2d(in_channels=n5x5_, out_channels=n5x5, kernel_size=5, padding=2),
                                 )
        
        self.Ix4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                 MainConv2d(in_channels=poolproj, out_channels=n1x1, kernel_size=1),)

    def forward(self, x):
        self.Iy1 = self.Ix1(x)
        self.Iy2 = self.Ix2(x)
        self.Iy3 = self.Ix3(x)
        self.Iy4 = self.Ix4(x)
        
        return torch.cat([self.Iy1, self.Iy2, self.Iy3, self.Iy4], 1)

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.7):
        super(AuxiliaryClassifier, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3, padding=2)
        self.conv1 = nn.Sequential(MainConv2d(in_channels=in_channels, out_channels=128, kernel_size = 1),
                                   nn.ReLU(True),)
        self.flatten_ = nn.Flatten()
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.dropout_ = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.flatten_(x)
        x = self.fc1(x)
        x = self.dropout_(x)
        x = self.fc2(x)
        
        return x

class GoogleNet_(nn.Module):
    def __init__(self, Aux=True):
        super(GoogleNet_, self).__init__()
        self.Aux = Aux

        self.mainConv_1 = MainConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.lr1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)

        self.mainConv_2 = MainConv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=1)
        self.mainConv_3 = MainConv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        
        self.lr2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
        
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3 , stride=2, padding=1)

        self.inception_1 = MainInception(in_channels=192, n1x1=64, n3x3_=96, n3x3=128, n5x5_=16, n5x5=32, poolproj=32)
        self.inception_2 = MainInception(in_channels=256, n1x1=128, n3x3_=128, n3x3=192, n5x5_=32, n5x5=96, poolproj=64)

        self.maxpool_3 = nn.MaxPool2d(kernel_size=3 , stride=2, padding=1)

        self.inception_3 = MainInception(in_channels=480, n1x1=192, n3x3_=96, n3x3=208, n5x5_=16, n5x5=48, poolproj=64)
        self.inception_4 = MainInception(in_channels=512, n1x1=160, n3x3_=112, n3x3=224, n5x5_=24, n5x5=64, poolproj=64)
        self.inception_5 = MainInception(in_channels=512, n1x1=128, n3x3_=128, n3x3=256, n5x5_=24, n5x5=64, poolproj=64)
        self.inception_6 = MainInception(in_channels=512, n1x1=112, n3x3_=144, n3x3=288, n5x5_=32, n5x5=64, poolproj=64)
        self.inception_7 = MainInception(in_channels=528, n1x1=256, n3x3_=160, n3x3=320, n5x5_=32, n5x5=128, poolproj=128)

        self.maxpool_4 = nn.MaxPool2d(kernel_size=3 , stride=2, padding=1)

        self.inception_8 = MainInception(in_channels=832, n1x1=256, n3x3_=160, n3x3=320, n5x5_=32, n5x5=128, poolproj=128)
        self.inception_9 = MainInception(in_channels=1024, n1x1=384, n3x3_=192, n3x3=384, n5x5_=48, n5x5=128, poolproj=128)

        self.avgpool_1 = nn.AvgPool2d(kernel_size=7, stride=1, padding=1)

        if self.Aux:
            self.auxiclas1 = AuxiliaryClassifier(in_channels=512, num_classes=1000, dropout=0.7)
            self.auxiclas2 = AuxiliaryClassifier(in_channels=528, num_classes=1000, dropout=0.7)

        self.dropout_1 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.mainConv_1(x)
        x = self.maxpool_1(x)

        x = self.lr1(x)

        x = self.mainConv_2(x)
        x = self.mainConv_3(x)

        x = self.lr2(x)

        x = self.maxpool_2(x)

        x = self.inception_1(x)
        x = self.inception_2(x)

        x = self.maxpool_3(x)

        x = self.inception_3(x)

        auxclas1 = self.auxiclas1(x)
        
        x = self.inception_4(x)
        x = self.inception_5(x)
        x = self.inception_6(x)

        x = self.auxiclas1(x)

        auxclas2 = self.inception_7(x)

        x = self.maxpool_4(x)

        x = self.inception_8(x)
        x = self.inception_9(x)

        x = self.avgpool_1(x)

        x = self.dropout_1(x)
        x = self.fc3(x)
        
        return x, auxclas1, auxclas2

transform = transforms.Compose([transforms.Resize((224,224)), 
                                transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor(), 
                                        transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.CIFAR10(root='./path', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./path' , train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

model = GoogleNet_()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(20):
    correct = 0
    total = 0
    total_loss = 0
    batch_loss = 0
    model.train()
    for images, lables in train_loader:
        images, lables = images.to(device), lables.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, lables)

        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += loss.item()

        y_pred = torch.argmax(output)
        correct += (y_pred == lables).sum().item()
        total += len(lables)

model.eval()

correct, total, batch_loss_eval, loss_eval = 0, 0, 0, 0

with torch.no_grad():
    for images_eval, lables_eval in test_loader:
        images_eval, lables_eval = images_eval.to(device), lables_eval.to(device)

        output = model(images_eval)
        loss = loss_fn(output, lables_eval)

        batch_loss_eval = loss.item()
        loss_eval += loss.item()

        y_pred_eval = nn.Softmax(output)
        y_pred_eval = torch.argmax(output)

        correct_eval += (y_pred_eval == lables).sum().item()
        total_eval += len(lables)

acc_eval = correct_eval/total_eval