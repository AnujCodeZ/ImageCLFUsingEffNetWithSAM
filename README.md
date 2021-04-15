# Image Classification Using EfficientNet and SAM.

## EfficientNet:
- It is a neural architecture search to design a new baseline network
and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets.

## SAM (Sharpness-Aware Minimization):
- It is used to generalize the model for better results on test set.
- Sharpness-Aware Minimization (SAM), find parameters that lie in neighborhoods
having uniformly low loss; this formulation results in a min-max optimization problem on which gradient descent can be performed efficiently.

> Using both can give state-of-the-art results. I implement them in a simple way on CIFAR-10 dataset.

## Usage:
`python3 train.py --version <version you want to use>`

- List of models:
    - 'b0'
    - 'b1'
    - 'b2'
    - 'b3'
    - 'b4'
    - 'b5'
    - 'b6'
    - 'b7'

## Refrences:
- Papers:
    - [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf)
    - [SAM](https://arxiv.org/pdf/2010.01412v2.pdf)
- Pytorch Codes:
    - [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)
    - [SAM](https://github.com/davda54/sam)
