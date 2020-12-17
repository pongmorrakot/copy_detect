import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

import numpy as np

from i3dBERT import *

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# weight_path ="./bert.weight"
root_path = "/media/ubuntu/Elements/ActivityNet/processed/"
label_path = "/media/ubuntu/Elements/ActivityNet/processed/_classLabel.txt"
train_path = "/media/ubuntu/Elements/ActivityNet/processed/_train_list.txt"
val_path = "/media/ubuntu/Elements/ActivityNet/processed/_val_list.txt"

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

class ANetDataset(Dataset):
    def __init__(self, classLabel, vidList, root_path=root_path):
        self.root_path = root_path
        self.vid_list = open(vidList, "r").readlines()
        self.label = open(classLabel, "r").readlines()
        self.class_num = len(self.label)

        vids = []
        for i in self.vid_list:
            vid_name, src, label, start, end = i.split()
            vids.append([self.root_path + vid_name + "/", int(label) - 1])
        self.vid_list = vids

    def import_frame(self,path):
        q = []
        for img in os.listdir(path):
            q.append(inpath + img)
        q.sort(key = lambda x: x.lower())

        seq = []
        for i in q:
            img = Image.open(i)
            img = transform(img)
            seq.append(img)
        return seq

    def target_gen(self, label):
        target = torch.zeros(self.class_num)
        target[label] = 1.
        return target

    def __getitem__(self, index):
        path, label = self.vid_list[index]
        # import all the frames
        input = self.import_frame(path)
        # generate target tensor from label
        target = self.target_gen(label)
        return input, target

    def __len__(self):
        return len(self.vid_list)


# TODO: add validation
def train(model, trainset, valset, loss_func=nn.CrossEntropyLoss, lr=0.001, epochs=20, batch_size=8):
    print("ViSiL Training")
    data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    running_loss = 0.0
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(data_loader):
            inputs, label = batch

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_func(outputs, label)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] Training loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                model.save_model()

        model.eval()
        with torch.no_grad():

            for i, batch in enumerate(val_loader):
                inputs, label = batch

                outputs = model(inputs)
                loss = loss_func(outputs, label)

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] Validation loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

trainset = ANetDataset(label_path, train_path)
valset = ANetDataset(label_path, val_path)
model = rgb_I3D64f_bert2(num_classes=trainset.class_num, extract=False).to(device)
train(model, trainset, valset)
