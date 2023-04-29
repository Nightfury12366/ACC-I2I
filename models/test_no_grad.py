import torch

a = torch.tensor([1., 1.], requires_grad=True)
b = a * 2


print(b)
b += 2


print(b)

