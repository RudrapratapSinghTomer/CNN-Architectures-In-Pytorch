import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from NNP.main_NNP import NNP_
from LeNet.main_LeNet import LeNet_
from AlexNet.main_AlexNet import AlexNet_
from VGGNet.main_VGG import VGG_
from GoogleNet.main_GoogleNet import GoogleNet_
from ResidualNet.main_ResNet34 import ResNet_34
from ResidualNet.main_ResNet50 import  ResNet_50
from MobileNet.main_MobileNet import MobileNetv1_

transform = transforms.Compose([transforms.Resize((224,224)), 
                                    transforms.RandomHorizontalFlip(), 
                                        transforms.ToTensor(), 
                                            transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.CIFAR10(root='./path', 
                                            train=True, 
                                                transform=transform, 
                                                    download=True)
test_dataset = datasets.CIFAR10(root='./path', 
                                            train=False, 
                                                transform=transform, 
                                                    download=True)

train_dataloader = DataLoader(dataset=train_dataset, 
                                            batch_size=16, 
                                                shuffle=True, 
                                                    batch_sampler=10)
test_dataloader = DataLoader(dataset=test_dataset, 
                                            batch_size=16, 
                                                shuffle=False, 
                                                    batch_sampler=10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_eval(model_class, train_dataloader, test_dataloader, device):
    # Initialize model
    model = model_class().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    model.train()
    for epoch in range(1):
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation Loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * (correct / total)

results = {}
models = {
    'NNP': NNP_, 
        'LeNet': LeNet_, 
            'AlexNet': AlexNet_,
                'VGG': VGG_,
                    'GoogleNet': GoogleNet_,
                        'ResNet34': ResNet_34, 
                            'ResNet50': ResNet_50
          }

for name, m_class in models.items():
    print(f"Training {name}...")
    acc = train_and_eval(m_class, train_dataloader, test_dataloader, device)
    results[name] = acc

# Plotting
# plt.bar(results.keys(), results.values())
# plt.ylabel("Accuracy (%)")
# plt.title('Model Accuracy Comparison')
# plt.show()