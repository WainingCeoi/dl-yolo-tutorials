import torch
import torch.nn as nn
import torch.nn.functional as F

# Create a Model Class that Inherits nn.Module
class Model(nn.Module):
    '''
    Input Layer (4 Features of Iris) --> Hidden Layer 1 (Number of Neurons) -->
    Hidden Layer 2 (Numeber of Neurons) --> Output (3 Classes of Iris)
    '''
    def __init__(self, input_features=4, h1=8, h2=9, output_features=3):
        super().__init__() # Instantiate our nn.Module
        self.fc1 = nn.Linear(input_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

# Pick a manual seed for randomization
torch.manual_seed(43)

# Create an instance of Model
model = Model()
