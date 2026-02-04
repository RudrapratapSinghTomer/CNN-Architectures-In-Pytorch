# CNN Architectures in PyTorch

This repository provides modular PyTorch implementations of various classic and modern Convolutional Neural Network (CNN) architectures. The goal is to evaluate and compare their performance (accuracy, training time, and complexity) using the CIFAR-10 dataset.

## 🚀 Overview
The project explores the evolution of CNNs—from early models like LeNet-5 to advanced architectures like EfficientNet. By increasing architectural complexity (depth, skip connections, and specialized convolutions), I will demonstrate how modern networks achieve superior results on image classification tasks.

## 🏗️ Supported Architectures
- [x] **Neural Network Perceptron (NNP)**: A basic baseline for comparison.
- [x] **LeNet-5**: The pioneer of CNNs, originally designed for handwritten digit recognition.
- [ ] **AlexNet**: The model that popularized deep learning in 2012. (Upcoming)
- [ ] **VGGNet**: Exploration of depth using small $3 \times 3$ filters. (Upcoming)
- [ ] **ResNet**: Utilizing skip connections to train ultra-deep networks. (Upcoming)
- [ ] **InceptionNet**: Multi-scale feature extraction through inception modules. (Upcoming)
- [ ] **MobileNet**: Lightweight convolutions for mobile devices. (Upcoming)
- [ ] **ShuffleNet**: Channel shuffling for efficient computation. (Upcoming)
- [ ] **EfficientNet**: Systematic scaling of depth, width, and resolution. (Upcoming)

NOTE: I main documentation tanh was used as activation funcation for LeNet however in this we are using RelU.

## 📊 Dataset
All models are trained and tested on the [CIFAR-10 Dataset](https://www.cs.toronto.edu), which consists of 60,000 32 \times 32 colour images in 10 classes.

## 📚 Documentation & References
- **LeNet-5 (Original Paper)**: [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu) by Yann LeCun et al.

- **LeNet-5 Guide**: [The Architecture of LeNet-5](https://www.analyticsvidhya.com) (Analytics Vidhya).

- **AlexNet Guide**: [Dive into Deep Learning](https://d2l.ai/chapter_convolutional-modern/alexnet.html)

- **VGGNet *Weight Layer 13* (Original Paper)**: [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556) by Karen Simonyan ∗ & Andrew Zisserman

- **VGGNet Guide**: [VGG-Net Architecture Explained](https://medium.com/@siddheshb008/vgg-net-architecture-explained-71179310050f), [Understand the Impact of Learning Rate on Neural Network Performance](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/)

- **GoogleNet (Original Paper)**: [Going Deeper with Convolutions](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf), [GoogLeNet](https://huggingface.co/learn/computer-vision-course/en/unit2/cnns/googlenet)

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

4. Performance Comparison
Comparison table will be updated as new models are implemented.
| Model | Parameters | Accuracy (%) | Epochs |
| :--- | :---: | :---: | :---: |
| LeNet-5 | ~60k | TBD | TBD |