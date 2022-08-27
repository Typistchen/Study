import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import visdom

# 初始化参数
learning_rate = 0.1
batch_size = 10
epochs = 10

# 数据加载
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size,
    shuffle=True
)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data',
        download=True,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.1307,))
        ])

    ),
    shuffle=True,
    batch_size=batch_size
)

# 网络搭建
class MLP(nn.Module):
    # 初始化继承，包含了相关参数的初始化
    def __init__(self):
        super(MLP, self).__init__()
        # 搭建网络
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 数据迁移至显卡
device = torch.device('cuda:0')
net = MLP().to(device) # 网络进行初始化

# 初始化优化器
optimizer = optim.SGD(net.parameters(), lr=learning_rate)  # 网络进行初始化值
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):   # batch_idx = train_loader / batch_size
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.cuda()

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item())
            )

