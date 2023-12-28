import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
# from models import *


class DNN_REC_Net(nn.Module):
    def __init__(self, input_size):
        super(DNN_REC_Net, self).__init__()
        self.hidden1 = nn.Linear(input_size, 512)
        self.hidden2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x), inplace=True)
        x = F.relu(self.hidden2(x), inplace=True)
        x = self.out(x).squeeze(dim=1)
        return torch.sigmoid(x) 