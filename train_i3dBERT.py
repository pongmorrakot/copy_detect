import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from i3dBERT import *

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

label_path = ""
train_path = ""
val_path = ""


class ANetDataset(Dataset):
    def __init__(self, classLabel, vidList):
        self.frame_entry = None
        self.label = None
        self.class_num = None

    def import_frame(self,path):
        q = []
        for img in os.listdir(path):
            q.append(inpath + img)
        q.sort(key = lambda x: x.lower())
        return q

    def target_gen(self, label):
        target = torch.zeros(self.class_num)
        target[label] = 1.
        return target

    def __getitem__(self, index):
        path, label = self.frame_entry[index]
        # import all the frames
        input = import_frame(path)
        # generate target tensor from label
        target = target_gen(label)
        return input, target

    def __len__(self):
        return len(self.frame_entry)


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
                    model.save_model()



model = rgb_I3D64f_bert2(weight_path, extract=False).to(device)
trainset = ANetDataset(label_path, train_path)
valset = ANetDataset(label_path, val_path)
train(model, trainset, valset)
