import torch

'''
Tensor Math Operations:
* Add, Subtract, Multiply, Divide, Remainders, Exponents
* Shorthand, Longhand
* Reassignment
'''

a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([5, 6, 7, 8])

# S1: Addition
print(a + b) # Shorthand
print(torch.add(a, b)) # Longhand 1
print(a.add(b)) # Longhand 2

# S2: Subtract Function
print(a - b)
print(torch.subtract(a, b))

# S3: Multiplication
print(a * b)
print(torch.multiply(a, b))

# S4: Division
print(a / b)
print(torch.divide(a, b))

# S5: Remainders Module
print(b % a)
print(torch.remainder(b, a))

# S6: Exponents / Power
print(a ** b)
print(torch.pow(a, b))

# S7: Reassignment with underscore
a = a + b # Recommended
print(a)
a += b # Recommended
print(a)
a = a.add_(b) # Not Recommended
print(a)
