import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

class GoogleNet_(nn.Module):
    def __init__():
        super().__init__()
        pass

    def forward_pass(self, x):
        pass

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.CIFAR10(root='./path', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root=,/path, train=False, transform=transform, download=True)

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

        output = model(images)
        loss = loss_fn(output, lables)

        loss.backward()
        optimizer.step()

        total_loss = loss.item()

        y_pred = torch.argmax(output)
        correct = (y_pred == lables).sum().item()
        total = len(lables)

model.eval()