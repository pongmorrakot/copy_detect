import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np

from evaluate import *

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

gt_path = "/home/ubuntu/Desktop/CC_WEB_Video/GT/"
label_path = "cc_web_video/cc_web_video.pickle"
weight_path = "./visil.weight"

class visil(nn.Module):
    def __init__(self, pretrained=None):
        # self.size = in_size
        # self.reLU = ) # applied after every Conv2D layer
        super(visil, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=[3, 3])
        self.mpool1 = nn.MaxPool2d([2, 2], 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=[3, 3])
        self.mpool2 = nn.MaxPool2d([2, 2], 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=[3, 3])
        self.fconv = nn.Conv2d(128, 1, kernel_size=[1, 1])

        if os.path.exists(pretrained):
            self.load_state_dict(torch.load(pretrained))

    def forward(self, x):
        x = nn.ReLU(self.conv1(x))
        x = self.mpool1(x)
        x = nn.ReLU(self.conv2(x))
        x = self.mpool2(x)
        x = nn.ReLU(self.conv3(x))
        x = self.fconv(x)
        return x


def triplet_generator_cc(class_index=-1):
    triplets = []

    gt = []
    if class_index == -1:
        for i in range(24):
            for t in open(gt_path +"GT_" + str(i+1) + ".rst", "r").readlines():
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
            pos.append(e[0])
        else:
            neg.append(e[0])


    features = load_rmac(class_index)
    for p in pos:
        for q in pos:
            for r in neg:
                anchor = features[retrieve_label(features, p)]
                positive = features[retrieve_label(features, q)]
                negative = features[retrieve_label(features, r)]
                print([anchor, positive, negative])
                triplets.append([anchor, positive, negative])
    return triplets


class FeatureDataset(Dataset):
    def __init__(self):
        self.features = triplet_generator_cc(1)

    def __getitem__(self, index):
        anchor, positive, negative = self.features[index]
        return anchor, positive, negative

    def __len__(self):
        return len(self.features)



def triplet_loss(model, anchor, positive, negative, margin=1.0):
    pos_sim = np.dot(anchor, positive.T)
    neg_sim = np.dot(anchor, negative.T)
    pos_sim = model(pos_sim)
    neg_sim = model(neg_sim)
    loss = pos_sim - neg_sim + margin
    if loss < 0.:
        loss = 0.
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

def train_visil(model, dataset, loss_func=nn.TripletMarginLoss(), lr=0.001, epochs=20, batch_size=8):
    print("ViSiL Training")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = optim.SGD(net.parameters(), lr=lr)
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

model = visil(pretrained=weight_path)
dataset = FeatureDataset()
train_visil = (model, dataset)
