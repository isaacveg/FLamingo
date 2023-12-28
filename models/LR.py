import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
# from models import *

class MNIST_LR_Net(nn.Module):
    def __init__(self):
        super(MNIST_LR_Net, self).__init__()
        self.hidden1 = nn.Linear(28 * 28, 512)
        self.hidden2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x), inplace=True)
        x = F.relu(self.hidden2(x), inplace=True)
        x = self.out(x)
        return F.log_softmax(x, dim=1)


class LR_Net(nn.Module):
    def __init__(self, input_size):
        super(LR_Net, self).__init__()
        self.hidden1 = nn.Linear(input_size, 1024)
        self.hidden2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.hidden1(x), inplace=True)
        x = F.relu(self.hidden2(x), inplace=True)
        x = self.out(x)
        return F.log_softmax(x, dim=1)