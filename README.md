# CNN Architectures in PyTorch

This repository provides modular PyTorch implementations of various classic and modern Convolutional Neural Network (CNN) architectures. The goal is to evaluate and compare their performance (accuracy, training time, and complexity) using the CIFAR-10 dataset.

## 🚀 Overview
The project explores the evolution of CNNs—from early models like LeNet-5 to advanced architectures like EfficientNet. By increasing architectural complexity (depth, skip connections, and specialized convolutions), I will demonstrate how modern networks achieve superior results on image classification tasks.

## 🏗️ Supported Architectures
- [x] **Neural Network Perceptron (NNP)**: A basic baseline for comparison.
- [x] **LeNet-5**: The pioneer of CNNs, originally designed for handwritten digit recognition.
- [X] **AlexNet**: The model that popularized deep learning in 2012.
- [X] **VGGNet**: Exploration of depth using small $3 \times 3$ filters.
- [X] **InceptionNet**: Multi-scale feature extraction through inception modules.
- [X] **ResNet**: Utilizing skip connections to train ultra-deep networks.
- [X] **MobileNet**: Lightweight convolutions for mobile devices. (In progress)
- [ ] **ShuffleNet**: Channel shuffling for efficient computation. (Upcoming)
- [ ] **EfficientNet**: Systematic scaling of depth, width, and resolution. (Upcoming)

NOTE: In main documentation tanh was used as activation funcation for LeNet however in this we are using RelU.

## 📊 Dataset
All models are trained and tested on the [CIFAR-10 Dataset](https://www.cs.toronto.edu), which consists of 60,000 32 \times 32 colour images in 10 classes.

## 📚 Documentation & References
- **LeNet-5 (Original Paper)**: [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu) by Yann LeCun et al.

- **LeNet-5 Guide**: [The Architecture of LeNet-5](https://www.analyticsvidhya.com) (Analytics Vidhya).

- **AlexNet Guide**: [Dive into Deep Learning](https://d2l.ai/chapter_convolutional-modern/alexnet.html)

- **VGGNet *Weight Layer 13* (Original Paper)**: [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556) by Karen Simonyan ∗ & Andrew Zisserman

- **VGGNet Guide**: [VGG-Net Architecture Explained](https://medium.com/@siddheshb008/vgg-net-architecture-explained-71179310050f), [Understand the Impact of Learning Rate on Neural Network Performance](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/)

- **GoogleNet (Original Paper)**: [Going Deeper with Convolutions](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)

- **GoogleNet Guide**: [GoogLeNet](https://huggingface.co/learn/computer-vision-course/en/unit2/cnns/googlenet), [GoogLeNet: A Deep Dive into Google’s Neural Network Technology](https://medium.com/@siddheshb008/googlenet-a-deep-dive-into-googles-neural-network-technology-f588d1b49e55)

- **Residual Net (Original Paper)**: [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)

- **ResNet18 Guide**: [Implementing ResNet18 in PyTorch from Scratch](https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/), [Wikipedia, Residual neural network](https://en.wikipedia.org/wiki/Residual_neural_network)

- **ResNet50 Guide**: [Building ResNets from Scratch using PyTorch](https://debuggercafe.com/building-resnets-from-scratch-using-pytorch/)

- **MobileNet (Original Paper)**: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision](https://arxiv.org/pdf/1704.04861), 

- **MobileNet Guide**:[Searching for MobileNetV3 | Paper Walkthrough & PyTorch Implementation](https://www.youtube.com/watch?v=0oqs-inp7sA&t=37s), [MobileNetV2 Paper Walkthrough: The Smarter Tiny Giant](https://towardsdatascience.com/mobilenetv2-paper-walkthrough-the-smarter-tiny-giant/)

- **UNet Guide**: [Implementing UNet from Scratch Using PyTorch](https://debuggercafe.com/unet-from-scratch-using-pytorch/), [Training UNet from Scratch using PyTorch](https://debuggercafe.com/training-unet-from-scratch/)

- **Pytorch Guide**: [Pytorch Documentation](https://docs.pytorch.org/docs/stable/torch.html),

## 🛠️ Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com

2. Install dependencies:
pip install -r requirements.txt

3. Run training
python main, NNP.py --model NNP_
python main, LeNet - 5.py --model LeNet_
python main, VGG.py -- model VGG_
python main, GoogleNet.py -- model GoogleNet_
python main, ResNet.py -- model ResNet_

4. Performance Comparison
Comparison table will be updated as new models are implemented.
| Model | Parameters | Accuracy (%) | Epochs |
| :--- | :---: | :---: | :---: |
| LeNet-5 | ~60k | TBD | TBD |