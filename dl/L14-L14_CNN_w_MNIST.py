import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix


# Convert MNIST Image files into a Tensor of 4-Dimensions: Number of Images, Height, Weight, Color Channels
transform = transforms.ToTensor()

# Train and Test Data
train_data = datasets.MNIST(root=r"./Datasets/MNIST", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=r"./Datasets/MNIST", train=False, download=True, transform=transform)
# Check with these Data
print(train_data)
print(test_data)
