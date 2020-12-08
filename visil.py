import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np
import random

from evaluate import *
from utils import pad

class visil(nn.Module):
    def __init__(self, weight_path):
        # self.size = in_size
        # self.reLU = ) # applied after every Conv2D layer
        super(visil, self).__init__()
        # self.conv1 = nn.Conv2d(2, 32, kernel_size=[3, 3])
        # self.mpool1 = nn.MaxPool2d([2, 2], 2)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=[3, 3])
        # self.mpool2 = nn.MaxPool2d([2, 2], 2)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=[3, 3])
        # self.fconv = nn.Conv2d(128, 1, kernel_size=[1, 1])

        self.conv1 = nn.Conv1d(512, 32, kernel_size=3)
        self.mpool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.mpool2 = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.fconv = nn.Conv1d(128, 1, kernel_size=1)

        self.weight_path = weight_path
        self.ReLU = nn.ReLU()

        if os.path.exists(self.weight_path):
            self.load_state_dict(torch.load(weight_path))
            self.eval()

    def forward(self, x):
        x = torch.from_numpy(x).to(device=device, dtype=torch.float)
        x = self.ReLU(self.conv1(x))
        x = self.mpool1(x)
        x = self.ReLU(self.conv2(x))
        x = self.mpool2(x)
        x = self.ReLU(self.conv3(x))
        x = self.fconv(x)
        return x

    def save_model(self):
        torch.save(self.state_dict(), self.weight_path)
        print("weight Saved")
