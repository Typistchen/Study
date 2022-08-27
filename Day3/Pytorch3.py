import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# 昨日练习
# x = torch.rand(1, 10)
# w = torch.rand(2, 10, requires_grad=True)
# out = x@w.t()
# print('out:', out)
# pred = F.sigmoid(out)
# label = torch.ones(1, 2)
# mse = F.mse_loss(pred, label)
# grad = torch.autograd.grad(mse, w)
# print('grad:', grad)

# 链式法则

# x = torch.tensor(1.)
# w1 = torch.tensor(2., requires_grad=True)
# b1 = torch.tensor(1.)
#
# w2 = torch.tensor(2., requires_grad=True)
# b2 = torch.tensor(1.)
#
# y1 = x * w1 + b1
# y2 = y1 * w2 + b2
#
# grad = torch.autograd.grad(y1, w1, retain_graph=True)[0]
# print('grad:', grad)
# grad2 = torch.autograd.grad(y2, y1, retain_graph=True)[0]
# grad3 = torch.autograd.grad(y2, w1)[0]
#
# grad3_2 = grad2 * grad
# print('grad3:', grad3)
# print('grad3_1:', grad3_2)

# 优化问题实战
def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
# print('x:', x)
print('x,y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y)
# print('X:', X)
print('X,Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y])

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

x = torch.tensor([4., 0.], requires_grad=True)
optimizer = torch.optim.Adam([x], lr=1e-3)
for step in range(20000):
    pred = himmelblau(x)

    optimizer.zero_grad()  # 清空梯度数据
    pred.backward()  # 反向传播，计算当前梯度
    optimizer.step()  # 根据梯度参数更新网络参数

    if step % 2000 == 0:
        print('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), pred.item()))

x = torch.rand(1, 784)
w = torch.rand(10, 784)
logits = x @ w.t()
loss = F.cross_entropy(logits, torch.rand(1, 10))  # 合并了softmax
print('crossentropy:', loss)

# 多分类问题

w1, b1 = torch.rand(200, 784, requires_grad=True), torch.rand(200, requires_grad=True)
w2, b2 = torch.rand(200, 200, requires_grad=True), torch.rand(200, requires_grad=True)
w3, b3 = torch.rand(10, 200, requires_grad=True), torch.rand(10, requires_grad=True)

def forward(x):
    x = x@w1.t() + b1
    x = F.relu(x)
    x = x@w2.t() + b2
    x = F.relu(x)
    x = x@w3.t() + b3
    x = F.relu(x)
    return x


x = torch.rand(10)
learning_rate = 1e-3
optimizer = torch.optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
