import os
import torch
import numpy as np
import pickle
import matplotlib.pylab as plt

from sklearn.metrics import precision_recall_curve

from feature_extract import pad_pair
from compute import sim, similarities
from utils import *

verbose = True

gt_path = "/home/ubuntu/Desktop/CC_WEB_Video/GT/"
label_path = "cc_web_video/cc_web_video.pickle"
ann_path = "/home/ubuntu/Desktop/CC_WEB_Video/CC_WEB_Video_List.txt"
rmac_feature_path = "./pca_features/"
i3d_feature_path = "./cc_web_video/"

def find_max(arr):
    cur_max = 0.
    max_index = 0
    for i,v in enumerate(arr):
        if v[1] > cur_max:
            cur_max = v[1]
            max_index = i
    return max_index

# data should be an array with each unit contains 2 elements: index in label files(video_id) and the data itself(ground_truth, feature, whatever)
def retrieve_label(data, label_index):
    for i,g in enumerate(data):
        if str(g[0]) == str(label_index):
            return i


def AP_eval(ground_truth, unsorted_sims, positive_labels='ESLMV'):
    # refer: the query video
    # ground_truth: label for each video in the class
    # unsorted_sims: scores of each video in the class
    i = 0.
    r = 0.
    s = 0.
    n = len(unsorted_sims)
    while unsorted_sims:
        index = find_max(unsorted_sims)
        label_index, value = unsorted_sims[index]
        if verbose:
            print(str(label_index) + "\t" + str(value) + "\t" + str(ground_truth[retrieve_label(ground_truth, label_index)]))
        r += 1
        if ground_truth[retrieve_label(ground_truth, label_index)][1] in positive_labels:
            i += 1
            s += i/r
        # print("i:\t" + str(i) + "\tr:\t" + str(r) + "\t" + str(i/r))
        unsorted_sims.pop(index)
    AP = s / i
    return AP


def load_i3d(class_index=-1):
    if class_index == -1:
        features = []
        for i in range(24):
            features += pickle.load(open(str(i+1) + '_extracted_features.pickle', 'rb'))
    else:
        features = pickle.load(open(str(class_index) + '_extracted_features.pickle', 'rb'))
    return features


def load_rmac(class_index=-1, inpath=rmac_feature_path, ann_path=ann_path):
    f_list = []
    # for f in folders:
    #     fpath = inpath + f + "/"
    #     vids = os.listdir(fpath)
    #     vids.sort(key=natural_keys)
    #     for v in vids:
    if class_index == -1:
        for entry in open(ann_path, "r"):
            items = entry.split()
            f_list.append([items[0], inpath + items[1] + "/" + items[3].split(".")[0]+".pkl"])
    else:
        for entry in open(ann_path, "r"):
            items = entry.split()
            if str(items[1]) == str(class_index):
                f_list.append([items[0], inpath + items[1] + "/" + items[3].split(".")[0]+".pkl"])
    if verbose:
        print("Video to load: %d" % len(f_list))
    features = []
    # load pickle files
    for f in f_list:
        features.append([f[0], pickle.load(open(f[1], 'rb'))])
        if verbose:
            print(f)
    if verbose:
        print("Feature #: %d " % len(features))
    # transpose
    final_features = []
    for f in features:
        final_features.append([f[0], np.transpose(f[1], (0,1))])
    return final_features


# for every class
# rank all the video in the query by scores
# iterate through every video: r increase as we iterate
# if the video is relavant: i++ and add i/r to the score

def eval_helper(cc_dataset, class_index, features, all_videos):
    q_index = cc_dataset['queries'][class_index-1]
    # q_index_in_list = q_index - 1
    if verbose:
        print("Index of Query/reference video: %d" % (q_index))

    # refer = features[q_index_in_list]
    query = features[retrieve_label(features, q_index)]
    if int(query[0]) != int(q_index):
        print("Query Index mismatch:\t%d\t%d" % (int(query[0]), int(q_index)))

    gt = []
    for t in range(len(features)):
        gt.append([t+1, 'X'])
    for t in open(gt_path +"GT_" + str(class_index) + ".rst", "r").readlines():
        truth = t.split()
        gt[retrieve_label(gt, truth[0])] = [truth[0], truth[1]]

    scores = []
    for f in features:
        scores.append([f[0],sim(query[1], f[1])])

    AP = AP_eval(gt,scores)
    print("Class " + str(class_index) + "\t" + str(AP))
    return AP


def eval_all(model="rmac"):
    mAP = 0.
    cc_dataset = pickle.load(open(cc_path, 'rb'))
    if model == "rmac":
        features = load_rmac()
    elif model == "i3d":
        features = load_i3d()
    if verbose:
        print("Feature imported: %d" % len(features))

    for i in range(24):
        class_index = i + 1
        AP = eval_helper(cc_dataset, class_index, features, all_videos=True)
        mAP += AP
    print("Final Score: "+ str(mAP/24))


def eval_class(class_index, all_videos, model="rmac"):
    cc_dataset = pickle.load(open(cc_path, 'rb'))

    if all_videos:
        if model == "rmac":
            features = load_rmac()
        elif model == "i3d":
            features = load_i3d()
    else:
        if model == "rmac":
            features = load_rmac(class_index)
        elif model == "i3d":
            features = load_i3d(class_index)

    if verbose:
        print("Feature imported: %d" % len(features))

    AP = eval_helper(cc_dataset, class_index, features, all_videos)
    return AP

# eval_class(1, False)
# eval_class(1, True)


# load a CC_WEB_VIDEO.pickle
# cc_dataset['queries'] : the index of what's considered the original video of each class
# cc_dataset['ground_truth'] : labels of every video in relation to its original
# cc_dataset['index'] : name of each videos

# calculate distance for every videos
#       for every video calculate distance between it and the query
# decide if the video is similar to any of the query video or whether it is dissimilar
# check the label to see if the video is actually similar
# cumulate mAP score
