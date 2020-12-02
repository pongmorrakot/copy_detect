import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


class visil(nn.Module):
    def __init__(self, pretrained=None):
        # self.size = in_size
        # self.reLU = ) # applied after every Conv2D layer

        self.conv1 = nn.ReLU(nn.Conv2d(2, 32, kernel_size=[3, 3]))
        self.mpool1 = nn.MaxPool2d([2, 2], 2)
        self.conv2 = nn.ReLU(nn.Conv2d(32, 64, kernel_size=[3, 3]))
        self.mpool2 = nn.MaxPool2d([2, 2], 2)
        self.conv3 = nn.ReLU(nn.Conv2d(64, 128, kernel_size=[3, 3]))
        self.fconv = tf.Conv2d(128, 1, kernel_size=[1, 1])

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained))

    def forward(self, x):
        x = self.conv1(x)
        x = self.mpool1(x)
        x = self.conv2(x)
        x = self.mpool2(x)
        x = self.conv3(x)
        x = self.fconv(x)
        return x


class VCDBdataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def generate_dataset(path):
        # extract feature and/or compute similarities, etc.
        pass


def train_visil(model, dataset, loss_func=nn.TripletMarginLoss(), lr=0.001, epochs=20, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    for epoch in range(epochs):
        for i, batch in enumerate(data_loader):
            anchor, positive, negative = data

            # zero the parameter gradients
            optimizer.zero_grad()

            #TODO: do something about this
            # forward + backward + optimize
            # outputs = model(inputs)
            # loss = loss_func(anchor, positive, negative)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

def
