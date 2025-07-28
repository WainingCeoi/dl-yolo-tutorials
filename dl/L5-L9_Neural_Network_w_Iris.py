import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as f
from tqdm.auto import  tqdm


'''Lecture 5: Create a Basic Neural Network'''
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
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.out(x)

        return x

# Pick a manual seed for randomization
torch.manual_seed(42)

# Create an instance of Model and put to Mac GPU, calls "mps"
model = Model()


'''Lecture 6: Load Data and Train Neural Network Model'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load iris datasets
iris = pd.read_csv(r"./Datasets/Iris.csv")
iris["variety"], uniques = pd.factorize(iris["variety"])

# Set X, y. Split train and test datasets. Convert into Tensor
# Default is float64, but tensor only support float32. Must convert to float 32 or will cause error
X = iris.drop(columns="variety").astype(np.float32).to_numpy()
# Default is int64, can convert to float32 or numpy int_, won't cause error if not make changes
y = iris["variety"].astype(np.int_).to_numpy()

# Method 1: Concisely, Recommended
X_train, X_test, y_train, y_test = [torch.tensor(data) for data in train_test_split(X, y, random_state=48)]

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Train our model
# Epochs: one run through all the training data in our network
epochs = 200
losses = []
# Use tqdm to show training progress
pbar = tqdm(range(epochs), desc="Training Progress", unit="epochs", file=sys.stdout)
for i in pbar:
    # Go forward and get a prediction
    y_prediction = model.forward(X_train)

    # Measure the train_loss/error, going to be high at first
    train_loss = criterion(y_prediction, y_train) # Predicted valves vs training values

    # Keep track of our losses
    losses.append(train_loss.item())

    # print each epoch loss
    pbar.set_postfix(loss=f"{train_loss:.4f}")

    # Do some back propagation: take the error rate of forward propagation and
    # feed it back through the network to find tune weights
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

# Graph it out
plt.plot(range(epochs), losses)
plt.ylabel("Loss/Error")
plt.xlabel("Epochs")
plt.title("Learning Curve")
plt.show()


'''Lecture 7: Evaluate Test Data Set on Network'''
# Evaluate Model on Test Data Set (Validate Model on Test Set)
with torch.no_grad(): # Basically turn off back propagation
    y_eval = model.forward(X_test) # X_test are feature from out test set, y_eval will be prediction
    test_loss = criterion(y_eval, y_test) # Find the train_loss or error
    print(test_loss)

correct = 0
with torch.no_grad():
    for idx, data in enumerate(X_test):
        y_val = model.forward(data)

        # Will tell us what type of flower class our network thinks it is
        '''
        Quick note for .item() Method:
        This method is used to extract the value from a single-element array or tensor as a standard Python scalar.
        It is particularly useful when the argmax() operation results in a single index
        e.g., from a 1D array or when finding the maximum across a specific dimension that 
        collapses the output to a single value.
        '''
        print(f"{idx+1}:\t{y_val} \t {uniques[y_val.argmax().item()]} \t {uniques[y_test[idx].item()]}")

        if y_val.argmax() == y_test[idx]:
            correct += 1

    print(f"We got {correct}/{len(y_test)} corrects!")


'''Lecture 8: New Data on the Network'''
new_iris1 = torch.tensor([4.7, 3.2, 1.3, 0.2])
with torch.no_grad():
    iris1_tensor = model.forward(new_iris1)
    print(f"{iris1_tensor} \t {uniques[iris1_tensor.argmax().item()]}")

new_iris2 = torch.tensor([5.9, 3.0, 5.1, 1.8])
with torch.no_grad():
    iris2_tensor = model.forward(new_iris2)
    print(f"{iris2_tensor} \t {uniques[iris2_tensor.argmax().item()]}")


'''Lecture 9: Save and Load our Neural Network Model'''
# Save our NN Model
torch.save(model.state_dict(), "Iris_NN_Model.pt")

# Load the saved Model
new_model = Model()
new_model.load_state_dict(torch.load("Iris_NN_Model.pt", weights_only=True))

# make sure it loaded correctly
print(new_model.eval())
new_iris3 = torch.tensor([4.8, 3.5, 1.2, 0.5])
with torch.no_grad():
    iris3_tensor = new_model.forward(new_iris3)
    print(f"{iris3_tensor} \t {uniques[iris3_tensor.argmax().item()]}")
