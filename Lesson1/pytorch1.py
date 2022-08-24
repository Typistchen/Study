import torch
a = torch.rand(2, 3)
print(a.type())
print('-----------')
print(type(a))
print('-----------')
# Dimension 0/rank 0 常用于Loss
b = torch.tensor(1.)
print(b)
print('-----------')
print(b.shape)
print('-----------')
print(len(b.shape))
print('-----------')

# Dimension 1/rank 1  Bias
c = torch.tensor([1.1])
print(c)
print('-----------')
d = torch.tensor([1.1, 2.2])
print(d)
print('-----------')
e = torch.FloatTensor(1)
print(e)