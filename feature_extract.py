import re
import os, sys, codecs
import subprocess
import math
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import pickle
from PIL import Image
from tqdm.notebook import tqdm
#import tqdm
import cv2
from sklearn.preprocessing import normalize as sknormalize
from sklearn.decomposition import PCA

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

import i3d
import vgg
# import compute
from extract_frame import extract
from utils import *


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


class QRDataset(Dataset):
    def __init__(self, img_path, transform = None):
        self.img_path = img_path

        self.img_label = np.zeros(len(img_path))

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])

        if self.transform is not None:
            img = self.transform(img)

        return img, self.img_path[index]

    def __len__(self):
        return len(self.img_path)


def normalize(x, copy = False):
    """
    A helper function that wraps the function of the same name in sklearn.
    This helper handles the case of a single column vector.
    """
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1, -1), copy = copy))
        #return np.squeeze(x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis])
    else:
        return sknormalize(x, copy = copy)
        #return x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis]


class Img2Vec():
    def __init__(self, model='i3d', layer='default', layer_output_size=512):
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if model == "i3d":
            self.model = i3d.rgb_I3D64f().to(device)
        elif model == "vgg16":
            self.model = vgg.vgg16.to(device)
            # take the bottom layer out
        else:
            print("What")
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def run(self, path):
        if not isinstance(path, list):
            path = [path]

        data_loader = torch.utils.data.DataLoader(QRDataset(path, self.transform), batch_size = 64,
                                                  shuffle = False, num_workers = 16)

        my_embedding = None

        with torch.no_grad():
            for index, batch_data in enumerate(data_loader):
                batch_data = batch_data[0].unsqueeze(0).transpose(1,2).to(device)
                # print(batch_data)
                # print("input shape:")
                # print(np.shape(batch_data))
                output = self.model(batch_data)
                # print("output shape:")
                # print(np.shape(output))
                if my_embedding is not None:
                	my_embedding = torch.cat((my_embedding, output), 2)
                else:
                    my_embedding = output
        # my_embedding = output
        # print(my_embedding)
        # print("Final output shape:\t")
        # print(np.shape(my_embedding))
        return my_embedding


def my_squeeze(arr):
    arr = arr.squeeze(4)
    arr = arr.squeeze(3)
    arr = arr.squeeze(0)
    return arr

def pad(arr, max_size):
    x,y = np.shape(arr)
    target = np.zeros((x, max_size))
    # # print(np.shape(target))
    # # print(np.shape(target[:,:y]))
    target[:,:y] = arr
    output = normalize(target)
    return output


def pad_pair(f1, f2):
    x1,y1 = np.shape(f1)
    x2,y2 = np.shape(f2)
    if y2 > y1:
        y_max = y2
    else:
        y_max = y1

    f1 = pad(f1, y_max)
    f2 = pad(f2, y_max)
    return f1,f2


def feature_extract(model, inpath):
    q = []
    for img in os.listdir(inpath):
        q.append(inpath + img)
    q.sort(key = lambda x: x.lower())
    print("feature extract:\tFrame_num:\t" + str(len(q)))

    x = model.run(q)
    x = my_squeeze(x)
    x = normalize(x.cpu())
    return x

def vid2vec(model, path, temp_path="./temp/"):
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    # extract frame
    extract(path, temp_path, 8)
    # feature_num = len(os.listdir(temp_path))
    # pass the path of said
    feature = feature_extract(model, temp_path)
    # print(np.shape(feature))
    feature_num = np.shape(feature)[1]
    # files = glob.glob("./temp/")
    shutil.rmtree(temp_path)
    return feature, feature_num


def extract_all(vid_path, ann_path, start_from=1):
    print('Extraction')
    print("Listing video")
    vid_list = []
    sz = []
    # f_path = vid_path + str(folder+1) + "/"
    # for v in os.listdir(f_path):
    #     vid_list.append(f_path + v)
    # vid_list.sort(key=natural_keys)
    for entry in open(ann_path, "r"):
        items = entry.split()
        if int(items[1]) >= start_from:
            vid_list.append([items[0], items[1], vid_path + items[1] + "/" + items[3]])
    for vid in vid_list:
        print(vid)
    print(len(vid_list))

    print("Start Running")
    features = []
    img2vec = Img2Vec(model="i3d")
    prev_class = str(start_from)
    for index, cur_class, vid in vid_list:
        if prev_class != cur_class:
            with open(str(prev_class) + '_extracted_features.pickle', 'wb') as pk_file:
                pickle.dump(features, pk_file, protocol = 4)
            print(str(prev_class) + "\tPickle file updated")
            features = []
        print(str(index) + "\tExtracting\t" + vid)
        feature, feature_num = vid2vec(img2vec, vid)
        print([index,np.shape(feature)])
        features.append([index,feature])
        prev_class = cur_class

    with open(str(prev_class) + '_extracted_features.pickle', 'wb') as pk_file:
        pickle.dump(features, pk_file, protocol = 4)
    print("Final Pre-pad Pickle file updated")


def pad_all():
    print("Start Padding")
    feature_sz = []
    for folder in range(24):
        features = pickle.load(open(str(folder+1) + '_extracted_features.pickle', 'rb'))
        for f in features:
            feature_sz.append(np.shape(f)[1])
        print(max(feature_sz))
    max_sz = max(feature_sz)
    padding = int(math.ceil(max_sz/100))*100
    print("Max frame num:\t" + str(max_sz) + "\tFeature Padding Size:\t" + str(padding))
    # padding = 1000

    for folder in range(24):
        features = pickle.load(open(str(folder+1) + '_extracted_features.pickle', 'rb'))
        new_features = []
        for i,f in enumerate(features):
            f = pad(f, padding)
            new_features.append(f)
            if i % 50 == 0:
                with open(str(folder+1) + '_extracted_features.pickle', 'wb') as pk_file:
                    pickle.dump(new_features, pk_file, protocol = 4)
                print(str(folder+1) + "\tPadded Pickle file updated")
        with open(str(folder+1) + '_extracted_features.pickle', 'wb') as pk_file:
            pickle.dump(new_features, pk_file, protocol = 4)
        print(str(folder+1) + "\tFinal Padded Pickle file updated")



vid_path = "/home/ubuntu/Desktop/CC_WEB_Video/video/"
ann_path = "/home/ubuntu/Desktop/CC_WEB_Video/CC_WEB_Video_List.txt"
# extract_all(vid_path, ann_path)
# pad_all()
