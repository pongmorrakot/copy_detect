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

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

gt_path = "/home/ubuntu/Desktop/CC_WEB_Video/GT/"
label_path = "cc_web_video/cc_web_video.pickle"
weight_path = "./visil.weight"

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


def triplet_generator_cc(class_index=-1):
    triplets = []

    gt = []
    if class_index == -1:
        for i in range(24):
            for t in open(gt_path +"GT_" + str(i+1) + ".rst", "positionalr").readlines():
                truth = t.split()
                gt.append([truth[0], truth[1]])
    else:
        for t in open(gt_path +"GT_" + str(class_index) + ".rst", "r").readlines():
            truth = t.split()
            gt.append([truth[0], truth[1]])
    pos = []
    neg = []

    for e in gt:
        if e[1] in 'ESLMV':
            print("pos\t" + str(e))
            pos.append(e[0])
        else:
            print("neg\t" + str(e))
            neg.append(e[0])


    features = load_rmac(class_index)
    size = []
    for f in features:
        size.append(np.shape(f[1])[0])
    for p in range(len(pos)):
        for q in range(p+1, len(pos)):
            anchor = features[retrieve_label(features, pos[p])]
            positive = features[retrieve_label(features, pos[q])]

            negative = features[retrieve_label(features, random.choice(neg))]
            # print([anchor, positive, negative])
            triplets.append([anchor, positive, negative])
    return triplets, max(size)


class FeatureDataset(Dataset):
    def __init__(self):
        self.features, self.size = triplet_generator_cc(1)
        # print(self.size)

    def __getitem__(self, index):
        anchor, positive, negative = self.features[index]
        # return pad(anchor, self.size), pad(positive, self.size), pad(negative, self.size)
        # print(anchor)
        # print(np.shape(anchor))
        # print(np.shape(positive))
        # print(np.shape(negative))
        # size = [np.shape(anchor[1])[0],np.shape(positive[1])[0],np.shape(negative[1])[0]]
        return pad(anchor[1].transpose(1,0),self.size), pad(positive[1].transpose(1,0), self.size), pad(negative[1].transpose(1,0), self.size)

    def __len__(self):
        return len(self.features)



def triplet_loss(model, anchor, positive, negative, margin=0.0):
    loss = 0.
    pos_sim = np.zeros((np.shape(anchor)[0],512,512))
    neg_sim = np.zeros((np.shape(anchor)[0],512,512))
    for i in range(np.shape(anchor)[0]):
        pos_sim[i] = np.dot(anchor[i], positive[i].T)
        neg_sim[i] = np.dot(anchor[i], negative[i].T)

    pos_sim = model(pos_sim)
    neg_sim = model(neg_sim)
    loss = pos_sim - neg_sim + margin
    # print(loss)
    loss = torch.mean(torch.max(loss, torch.zeros_like(loss)))
    return loss

# From ViSiL, for reference
def similarity_regularization_loss(sim, lower_limit=-1., upper_limit=1.):
    with tf.variable_scope('similarity_regularization_loss'):
        return tf.reduce_sum(tf.abs(tf.minimum(.0, sim - lower_limit))) + tf.reduce_sum(tf.abs(tf.maximum(.0, sim - upper_limit)))

# sim : the similarity matrix
# label : 0 or 1, indicate if the matrix suppose to be similar or dissimilar
# # TODO: This is most likely wrong I need to
# def visil_loss(sim, label):
#     x,y = np.shape(sim)
#     gt = np.zeros((x,y))
#     if label == 1:
#         gt = np.ones((x,y))
#     loss = gt - sim
#     return loss

def train_visil(model, dataset, loss_func=triplet_loss, lr=0.001, epochs=20, batch_size=8):
    print("ViSiL Training")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    running_loss = 0.0
    for epoch in range(epochs):
        for i, batch in enumerate(data_loader):
            anchor, positive, negative = batch

            # zero the parameter gradients
            optimizer.zero_grad()

            loss = loss_func(model, anchor, positive, negative)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                model.save_model()

model = visil(weight_path).to(device)
dataset = FeatureDataset()
train_visil(model, dataset)
