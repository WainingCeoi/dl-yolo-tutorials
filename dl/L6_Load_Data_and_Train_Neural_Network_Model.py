import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create a Model Class that Inherits nn.Module
class Model(nn.Module):
    """
    Input Layer (4 Features of Iris) --> Hidden Layer 1 (Number of Neurons) -->
    Hidden Layer 2 (Number of Neurons) --> Output (3 Classes of Iris)
    """
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

# Create an instance of Model and put to Mac GPU, calls "mps"
model = Model().to("mps")

# --- Above Code are Copied from Last Lecture ---
# --- Below Code are for this Lecture ---
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris datasets
iris = load_iris()

# Set X, y. Split train and test datasets. Convert into Tensor
X = iris.data.astype(np.float32)
y = iris.target.astype(np.int_)

# Method 1: Concisely, Recommended
X_train, X_test, y_train, y_test = [torch.tensor(data, device="mps") for data in train_test_split(X, y)]

'''
Method 2: Originally, Not Recommended

Step 1: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y)

Step2 : Convert Datasets into Tensor
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)
'''


# Set the criterion of model to measure the error, how far off the predictions are from the data
criterion = nn.CrossEntropyLoss()
# Choose Adam optimizer, lr = learning rate
# If error doesn't go down after a bunch of iteration (epochs), lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train our model
# Epochs: one run thru all the training data in our network
epochs = 200
losses = []
for i in range(epochs):
    # Go forward and get a prediction
    y_pred = model.forward(X_train)

    # Measure the loss/error, gonna be high at first
    loss = criterion(y_pred, y_train).to("cpu") # Predicted valves vs training values

    # Keep track of our losses
    losses.append(loss.detach().numpy())

    # print each epoch
    if (i+1) % 10 == 0:
        print(f"Epoch: {i+1}, loss: {loss:.4f}")

    # Do some back propagation: take the error rate of forward propagation and
    # feed it back thru the network to find tune weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Graph it out
plt.plot(range(epochs), losses)
plt.ylabel("Loss/Error")
plt.xlabel("Epochs")
plt.title("Learning Curve")
plt.show()
