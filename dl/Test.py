import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(10, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

y = torch.rand(100, 50)
print(y)
