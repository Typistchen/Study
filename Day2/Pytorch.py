import torch
import torch.nn
import torch.nn.functional as F
# 梯度
a = torch.linspace(-100, 100, 10)

print(a)

b = torch.sigmoid(a)
print(b)
c = torch.tanh(a)
print(c)
d = torch.relu(a)
print(d)

# 均方差求梯度
x = torch.ones(1)   # 生成一个torch类型的x的初始值 初始值为1
w = torch.full([1], 2.0, requires_grad=True)  # 生成一个torch类型的y的初始值 初始值为2
# w.requires_grad_()
label = torch.ones(1)  # 标签值为label
mse = F.mse_loss(label, x * w)  #

grad = torch.autograd.grad(mse, [w])
print(grad)

f = torch.rand(3, requires_grad=True)
pred2 = F.softmax(f, dim=0)  # y = softmax(x)  y1 = softmax(x1)
# label2 = torch.ones(3)
# mse2 = F.mse_loss(label2, pred2)
grad2 = torch.autograd.grad(pred2[0], f)  # 只能对y1进行求导，所以前面只能用单个的数
print(pred2)

# 单个感知机
x = torch.rand(1, 10)
w = torch.rand(1, 10, requires_grad=True)
pred3 = torch.sigmoid(x@w.t())
print('x@w.t():', x@w.t())
print(pred3)
label1 = torch.ones(1, 1)
mse2 = F.mse_loss(label1, pred3)
print(mse2)
grad3 = torch.autograd.grad(mse2, w)
print(grad3)

# 多维度输出感知机
x = torch.rand(1, 10)
w = torch.rand(2, 10, requires_grad=True)
out1 = x@w.t()
print('out1:', out1)
label3 = torch.rand(1, 2)
mse3 = F.mse_loss(label3, out1)
print(mse3)
grad4 = torch.autograd.grad(mse3, w)
print('grad4:', grad4)
