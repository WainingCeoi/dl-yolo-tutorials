import numpy as np
import torch

# S1
my_torch = torch.arange(10)
print(my_torch)

# S2: Reshape and view
'''
In summary differences between shape and view:
1. Use view() when you are certain the tensor is contiguous and you desire memory sharing
   (i.e., you want changes to reflect in both the original and reshaped tensor).
2. Use reshape() when you need to change the shape regardless of contiguity 
   and are less concerned about strict memory sharing, as it might create a copy.
   If you are unsure about the contiguity of your tensor,
   reshape() is generally the safer choice as it will not raise an error.
'''
print(my_torch.reshape([2, 5]))
print(my_torch.view([2, 5]))

# S3: Reshape if we don't know the number of items using -1
my_torch2 = torch.arange(15)

print(my_torch2.reshape([3, -1]))
print(my_torch2.reshape([-1, 3]))


# S4: With reshape and view, relative tensors will update if original tensors make changes. (Also apply to NumPy Array)
my_torch3 = torch.arange(10)
my_torch4 = my_torch3.reshape([2, 5])
print(my_torch3)
print(my_torch4)

my_torch3[1] = 123
print(my_torch3)
print(my_torch4)

# S5: Slices
my_torch5 = torch.arange(15)
# Grab a specific item
print(my_torch5[7])
# Grab slice
my_torch6 = my_torch5.reshape([5, 3])

# Return in a row, cause [:, 1] mean pick all row and pick column idx=1 item
print(my_torch6[:, 1])

# Return in a column, cause [:, 1:2] mean pick all row and pick column form idx=1 to idx=1,
# number after colon will not be included
print(my_torch6[:, 1:2])
