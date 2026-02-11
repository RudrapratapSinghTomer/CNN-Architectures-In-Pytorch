import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from NNP import NNP_
from LeNet import LeNet_
from AlexNet import AlexNet_
from VGGNet import VGG_
from GoogleNet import GoogleNet_
from ResidualNet import ResNet_34, ResNet_50
from MobileNet import MobileNet_

transform = transforms.Compose([transforms.Resize((256,256)), 
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cup')

def evaluatin(model, test_dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad:
        for images, lables in test_dataloader:
            images, lables = images.to(device), lables.to(device)

            output = model(images)

            _, y_pred = torch.max(output.data, 1)
            total += lables.size(0)
            correct += (y_pred == lables).sum().item()

    return 100 * (correct/total)

results = {}
models = {'NNP': NNP_, 
          'LeNet': LeNet_, 
          'AlexNet': AlexNet_,
          'VGG': VGG_,
          'GoogleNet': GoogleNet_,
          'ResNet34': ResNet_34, 
          'ResNet50': ResNet_50}

for name, model in models:
    results[name] = evaluatin(model=model, test_dataloader=test_dataset, device=device)

#plot
plt.bar(results.keys(), results.values(), color=['blue', 'green', 'red', 'purple'])
plt.ylabel("accuracy(%)")
plt.title('Model Accuracy Comparison')
plt.show()