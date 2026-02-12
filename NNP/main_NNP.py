import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class NNP_(nn.Module):
    def __init__(self):
        super(NNP_, self).__init__()
        fc1 = nn.Linear(16*4*4,120)
        fc2 = nn.Linear(120,84)
        fc3 = nn.Linear(84,10)

    def forward_pass(self, x):

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

transform = transforms.Compose([
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

train_dataloader = DataLoader(train_dataset, 1000, batch_size=64, shuffle=True)

test_dataloader = DataLoader(test_dataset, 1000, batch_size=64, shuffle=False)

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

model = NNP_().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters, lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0 #
    correct = 0 #
    total = 0 #

    for images, labels in train_dataloader:
        images, lables = images.to(device), labels.to(device)
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