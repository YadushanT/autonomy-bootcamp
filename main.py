"""
This is a starter file to get you going. You may also include other files if you feel it's necessary.

Make sure to follow the code convention described here:
https://github.com/UWARG/computer-vision-python/blob/main/README.md#naming-and-typing-conventions

Hints:
* The internet is your friend! Don't be afraid to search for tutorials/intros/etc.
* We suggest using a convolutional neural network.
* TensorFlow Keras has the CIFAR-10 dataset as a module, so you don't need to manually download and unpack it.
"""

# Import whatever libraries/modules you need
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot
from keras.datasets import cifar10
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler


# Your working code here

#check availability of CUDA
trainGPU = torch.CUDA.is_available()

if not trainGPU:
    print("CUDA not available. Training on CPU")
else:
    print("CUDA available, training on GPU")

