import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class LeNet_(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.fc1 = nn.Linear(400,120) #need to update this later
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

transform = transforms.Compose([transforms.Resize((32,32)),
                                transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.CIFAR10(root="./data", 
                                                train=True, 
                                                    download=True, 
                                                        transform=transform)

test_dataset = datasets.CIFAR10(root="./data", 
                                                train=False, 
                                                    download=True, 
                                                        transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

model = LeNet_().to(device)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total_loss = 0 #
    correct = 0 #
    total = 0 #

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        y_pred = torch.argmax(output, dim=1)
        correct += (y_pred == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}, Loss={total_loss/len(train_dataloader):.4f}, Acc={acc:.4f}")

model.eval()

corrects = 0
totals = 0

with torch.no_grad():
    for images, labels in test_dataloader:
        images, lables = images.to(device), labels.to(device)
        output = model(images)

        y_pred = torch.softmax(output, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        correct += (y_pred == labels).sum().item()
        total += labels.sum().item()

print("Test Accuracy:", corrects / total)