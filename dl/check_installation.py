import torch

# Check if PyTorch is installed
x = torch.rand(3, 4)
print(x)

# Check if MPS available.
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    y = torch.ones(10, device=mps_device)
    print (y)
else:
    print ("MPS device not found.")
