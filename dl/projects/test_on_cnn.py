import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# Build CNN Model
class CNN (nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1)
        self.conv2 = nn.LazyConv2d(out_channels=16, kernel_size=3, stride=1)

        self.fc1 = nn.LazyLinear(out_features=128)
        self.fc2 = nn.LazyLinear(out_features=96)
        self.fc3 = nn.LazyLinear(out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x


# Initial and Read Model
model = CNN()
model.load_state_dict(torch.load(r"../models/MNIST.pt"))

# Collect Hand Write Number Images
image_folder = r"../datasets/MNIST/hand_writing"
images = os.scandir(image_folder)
correct = 0

for img in images:
    # Load Images
    img_name = int(img.name.replace(".jpg", ""))
    img = Image.open(img.path).resize((28, 28)).convert("L")

    plt.imshow(img, cmap="Greys")
    plt.show()

    # Convert Image into Tensor
    img_array = (255-np.array(img))
    img_tensor = torch.Tensor(img_array).reshape(1, 1, 28, 28)

    # Send Image through Model
    with torch.no_grad():
        predicted_number = model(img_tensor).argmax()
        correct += (img_name == predicted_number).sum()
        print(f"Label: {img_name}, Predicted: {predicted_number.item()}, {img_name == predicted_number}")

print(f"We got {correct}/{len(os.listdir(image_folder))}!")
