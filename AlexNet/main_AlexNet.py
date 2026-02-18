import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class AlexNet_(nn.Module):
    def __init__(self):
        super().__init__()
        #changing kernel size to 3, stride to 1, and padding to 1 for 32x32 input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)

        self.fc1 = nn.Linear(4*4*256,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = F.max_pool2d(F.relu(self.conv5(x)), 2)

        x = torch.flatten(x, 1)

        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5)

        x = self.fc3(x) 

        return x

transform = transforms.Compose([
    #transforms.Resize((224,224)), #change this as per preferance, for now 32 to save memory
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), 
                         (0.5,0.5,0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True) # Added pin_memory=True to help the GPU pipeline
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

model = AlexNet_().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, lables in train_dataloader:
        images , lables = images.to(device), lables.to(device)
        output = model(images)
        loss = loss_fn(output, lables)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        y_pred = torch.argmax(output, dim=1)
        correct += (y_pred == lables).sum().item()
        total += lables.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}, Loss={total_loss/len(train_dataloader):.4f}, Acc={acc:.4f}")