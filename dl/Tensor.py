import torch
import numpy as np

'''
Quick Notes for Tensors:
* A torch. Tensor is a multi-dimensional matrix containing elements of a single data type.
* Similar to NumPy Arrays, but full of fun things that make them work better on GPU's (vs regula CPU's)
* Default data type of float 32.
* More suitable for deep learning than a numpy array.
'''

# Python Lists
a = [1, 2, 3, 4, 5]

# NumPy Arrays
x = np.random.rand(2, 3, 4)
print(x)

# Tensors
y = torch.rand(2, 3, 4)
print(y)

# Convert NumPy arrays to tensors
z = torch.tensor(x)
print(z)
