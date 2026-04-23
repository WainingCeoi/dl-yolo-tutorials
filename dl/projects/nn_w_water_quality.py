import sys
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm


df = pd.read_csv(r"../datasets/water_quality.csv").dropna()

assert df.isna().sum().sum() == 0
X = df.drop(columns="Potability").astype(np.float32).to_numpy()
y = df["Potability"].astype(np.int_).to_numpy()
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
X_train, X_test, y_train, y_test = (torch.tensor(data).to(device) for data in train_test_split(X, y))

class fc_model(nn.Module):
    def __init__ (self):
        super().__init__()
        self.fc1 = nn.LazyLinear(out_features=12)
        self.fc2 = nn.LazyLinear(out_features=2)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


model = fc_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizor = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 1000
train_losses = []
train_accuracies = []
pbar = tqdm(range(epochs), desc="Training Progress", unit="epochs", file=sys.stdout)
t0 = time()
for idx in pbar:
    y_pred = model(X_train)

    loss = criterion(y_pred, y_train)
    train_losses.append(loss.item())

    correct = 0
    correct = (y_pred.argmax(dim=1) == y_train).sum()
    accuracy = 100 * correct / len(y_train)
    train_accuracies.append(accuracy)

    optimizor.zero_grad()
    loss.backward()
    optimizor.step()

    run_time = time() - t0
    pbar.set_postfix({"Loss": f"{loss:.4f}", "Time": f"{run_time:.2f}s", "Accuracy": f"{accuracy:.2f}%"})
