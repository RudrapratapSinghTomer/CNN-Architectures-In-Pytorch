import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class VGG_(nn.Module):
    def __init__(self):
        super(VGG_).__init__()
        conv11 = nn.Conv2d(3, 64, 3, padding=1, stride=1)
        conv12 = nn.Conv2d(64, 64, 3, padding=1, stride=1)

        conv21 = nn.Conv2d(64, 128, 3, padding=1, stride=1)
        conv22 = nn.Conv2d(128, 128, 3, padding=1, stride=1)

        conv31 = nn.Conv2d(128, 256, 3, padding=1, stride=1)
        conv32 = nn.Conv2d(256, 256, 3, padding=1, stride=1)

        conv41 = nn.Conv2d(256, 512, 3, padding=1, stride=1)
        conv42 = nn.Conv2d(512, 512, 3, padding=1, stride=1)

        conv51 = nn.Conv2d(512, 512, 3, padding=1, stride=1)
        conv52 = nn.Conv2d(512, 512, 3, padding=1, stride=1)

        fc1 = nn.Linear(25088, 4096)
        fc2 = nn.Linear(4096, 4096)
        fc3 = nn.Linear(4096, 1000)

    def forward_pass(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = nn.MaxPool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = nn.MaxPool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = nn.MaxPool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv41(x))
        x = F.relu(self.conv42(x))
        x = nn.MaxPool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv51(x))
        x = F.relu(self.conv52(x))
        x = nn.MaxPool2d(x, kernel_size=2, stride=2)

        x = nn.Flatten(x, 1)

        x = F.relu(self.fc1(x))
        nn.Dropout(p=0.5)

        x = F.relu(self.fc2(x))
        nn.Dropout(p=0.5)

        x = F.relu(self.fc3(x))
        nn.Dropout(p=0.5)

        return x

transform = transforms.Compose([
                            transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.CIFAR10(root='./data',
                                 train=True,
                                transform=transform,
                                download=True)

test_dataset = datasets.CIFAR10(root='./data',
                                train=False,
                                transform=transform,
                                download=True)

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

device = torch.device(f'cuda' if torch.cuda.is_available else 'cpu')

model = VGG_().to(device=device)
model.to(device)
optimizer = optim.SGD(model.get_parameter, lr=0.01, momentum=0.9, weight_decay=0.0005)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(74):
    correct = 0
    total = 0
    total_loss = 0
    model.train()

    for images, lables in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, lables)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        y_pred = torch.argmax(output, dim=1)
        correct += (y_pred == lables).sum().item()
        total += labels.size()

    acc = correct/total
    print(f"Epoch {epoch+1}, Loss={total_loss/len(train_dataloader):.4f}, Acc={acc:.4f}")