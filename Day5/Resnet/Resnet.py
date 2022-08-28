import torch
from torch import  nn
import torch.nn.functional as F

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

