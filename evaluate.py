import os
import torch
import numpy as np
import pickle
import matplotlib.pylab as plt

from sklearn.metrics import precision_recall_curve

from feature_extract import pad_pair
from compute import sim, similarities
from utils import *

label_path = 'cc_web_video.pickle'
feature_path = '_extracted_features.pickle'

verbose = True

# use for references
def calculate_similarities(queries, features, metric='euclidean'):
    """
      Function that generates video triplets from CC_WEB_VIDEO.
      Args:
        queries: indexes of the query videos
        features: global features of the videos in CC_WEB_VIDEO
      Returns:
        similarities: the similarities of each query with the videos in the dataset
    """
    similarities = []
    # for every video calculate its distance with all(24) of the query all_videos
    dist = np.nan_to_num(cdist(features[queries], features, metric=metric))

    for i, v in enumerate(queries):
        sim = np.round(1 - dist[i] / dist.max(), decimals=6)
        similarities += [[(s, sim[s]) for s in sim.argsort()[::-1] if not np.isnan(sim[s])]]
    return similarities

def evaluate(ground_truth, similarities, positive_labels='ESLMV', all_videos=False):
    """
      Function that plots the PR-curve.
      Args:
        ground_truth: the ground truth labels for each query
        similarities: the similarities of each query with the videos in the dataset
        positive_labels: labels that are considered positives
        all_videos: indicator of whether all videos are considered for the evaluation
        or only the videos in the query subset
      Returns:
        mAP: the mean Average Precision
        ps_curve: the values of the PR-curve
    """
    pr, mAP = [], 0.0
    # going through every video's label
    for query_set, labels in enumerate(ground_truth):
        i = 0.0
        ri = 0
        s = 0.0
        y_target, y_score = [], []

        # iterate through similarity score with each class of video
        for video, sim in similarities[query_set]:
            if all_videos or video in labels:
                y_score += [sim]
                y_target += [0.0]
                ri += 1
                if video in labels and labels[video] in positive_labels:
                    i += 1.0
                    s += i / ri
                    y_target[-1] = 1.0

        mAP += s / np.sum([1.0 for label in labels.values() if label in positive_labels])

        precision, recall, thresholds = precision_recall_curve(y_target, y_score)
        p = []
        for i in xrange(20, 0, -1):
            idx = np.where((recall >= i*0.05))[0]
            p += [np.max(precision[idx])]
        pr += [p + [1.0]]

    return mAP / len(ground_truth), np.mean(pr, axis=0)[::-1]


# def eval(refer, query, thres=0):
#     scores = []
#     for r in refer:
#         score = sim(test_feature, r)
#         scores.append(score)
#     index = np.argmax(scores)
#     if scores[index] > thres:
#         return index + 1, scores[index]
#     else:
#         return -1, scores[index]
#
# def correct_class(refer_index, cur_index):
#     for i, r in enumerate(refer_index):
#         if cur_index <= r - 1:
#             return i + 1
#     return -1

def find_max(arr):
    cur_max = 0.
    max_index = 0
    for i,v in enumerate(arr):
        if v[1] > cur_max:
            cur_max = v[1]
            max_index = i
    return max_index

def retrieve_label(gt, label_index):
    for i,g in enumerate(gt):
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
        # print(str(label_index) + "\t" + str(value) + "\t" + str(ground_truth[retrieve_label(ground_truth, label_index)]))
        r += 1
        if ground_truth[retrieve_label(ground_truth, label_index)][1] in positive_labels:
            i += 1
            s += i/r
        # print("i:\t" + str(i) + "\tr:\t" + str(r) + "\t" + str(i/r))
        unsorted_sims.pop(index)
    AP = s / i
    return AP



# generate refer_features; if not already generated
def generate_refer_i3d():
    print("load data")
    cc_dataset = pickle.load(open('cc_web_video.pickle', 'rb'))
    refer = []
    q0 = 0
    for i in range(24):
        print(cc_dataset['queries'][i])
        print(q0)
        print(cc_dataset['queries'][i]-q0-1)
        features = pickle.load(open(str(i+1) + '_extracted_features.pickle', 'rb'))
        refer.append(features[cc_dataset['queries'][i]-q0-1])
        q0 += len(features)
    with open("refer_features.pickle", 'wb') as pk_file:
        pickle.dump(refer, pk_file, protocol = 4)
    print("done")


def load_i3d(class_index):
    features = pickle.load(open(str(class_index) + '_extracted_features.pickle', 'rb'))
    return features


def generate_refer_rmac():
    print("load data")
    cc_dataset = pickle.load(open('cc_web_video/cc_web_video.pickle', 'rb'))
    refer_list = cc_dataset['queries']
    refer = []
    if class_index == -1:
        for entry in open(ann_path, "r"):
            items = entry.split()
            if int(items[0]) in refer_list:
                refer.append(pickle.load(open(inpath + items[1] + "/" + items[3].split(".")[0]+".pkl", 'rb')))
    with open("refer_features.pickle", 'wb') as pk_file:
        pickle.dump(refer, pk_file, protocol = 4)
    print("done")


def load_rmac(class_index=-1, inpath="./pca_features/", ann_path="/home/ubuntu/Desktop/CC_WEB_Video/CC_WEB_Video_List.txt"):
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
# for n,t in cc_dataset['ground_truth'][0].items():
# for t in open("/home/ubuntu/Desktop/CC_WEB_Video/GT/GT_" + str(1) + ".rst", "r").readlines():
#     print(t.split()[1])

# mAP
# def eval_all():
#     print("load data")
#     cc_dataset = pickle.load(open('cc_web_video/cc_web_video.pickle', 'rb'))
#     refer = pickle.load(open('refer_features.pickle', 'rb'))
#     num_classes = 24
#     mAP = 0.
#     for i in range(num_classes):
#         if model == "i3d":
#             features = load_i3d(i+1)
#         elif model == "rmac":
#             features = load_rmac(i+1)
#         gt = []
#         for t in open("/home/ubuntu/Desktop/CC_WEB_Video/GT/GT_" + str(i+1) + ".rst", "r").readlines():
#             truth = t.split()
#             gt.append([truth[0], truth[1]])
#
#         # print(len(features))
#         # print(len(gt))
#         # # check if there is mismatch
#         for index in range(len(features)):
#             if str(features[index][0]) != str(gt[index][0]):
#                 print("Mismatch found at:\t" + features[index][0] + "\t" + gt[index][0])
#
#         scores = []
#         for f in features:
#             scores.append([f[0],sim(refer[i], f[1])])
#
#         # print(len(scores))
#         # print(len(gt))
#         AP = AP_eval(gt,scores)
#         print("Class " + str(i+1) + "\t" + str(AP))
#         mAP += AP
#     print("Final score:(not sure if this is how you calculate final score)")
#     print(mAP / num_classes)

def eval_helper(cc_dataset, class_index, features):
    q_index = cc_dataset['queries'][class_index-1]
    refer_index = q_index - 1
    if verbose:
        print("Index of Query/reference video: %d %d" % (refer_index, q_index))

    refer = features[refer_index]
    if int(refer[0]) != int(q_index):
        print("Query Index mismatch:\t%d\t%d" % (int(refer[0]), int(q_index)))
    scores = []

    gt = []
    for t in range(len(features)):
        gt.append([t+1, 'X'])
    for t in open("/home/ubuntu/Desktop/CC_WEB_Video/GT/GT_" + str(class_index) + ".rst", "r").readlines():
        truth = t.split()
        gt[int(truth[0])-1] = [truth[0], truth[1]]

    for f in features:
        scores.append([f[0],sim(refer[1], f[1])])
    AP = AP_eval(gt,scores)
    print("Class " + str(class_index) + "\t" + str(AP))
    return AP


def eval_all():
    mAP = 0.
    cc_dataset = pickle.load(open('cc_web_video/cc_web_video.pickle', 'rb'))
    features = load_rmac()
    if verbose:
        print("Feature imported: %d" % len(features))

    for i in range(24):
        class_index = i + 1
        AP = eval_helper(cc_dataset, class_index, features)
        mAP += AP
    print("Final Score: %d" % mAP/24)


def eval_class(class_index, all_videos=True):
    cc_dataset = pickle.load(open('cc_web_video/cc_web_video.pickle', 'rb'))

    if all_videos:
        features = load_rmac()
    else:
        features = load_rmac(class_index)
    if verbose:
        print("Feature imported: %d" % len(features))

    AP = eval_helper(cc_dataset, class_index, features)
    return AP


# features = []
# for i in range(24):
#     features += load_rmac(i+1)
#     print(str(i+1) + "\tloaded")

# for i in range(24):
#     eval_class(i+1)

eval_all()


# inpath= "./pca_features/1/1_4_Y.pkl"
# feature = pickle.load(open(inpath, 'rb'))
# print(np.shape(feature))





# load a CC_WEB_VIDEO.pickle
# cc_dataset['queries'] : the index of what's considered the original video of each class
# cc_dataset['ground_truth'] : labels of every video in relation to its original
# cc_dataset['index'] : name of each videos

# calculate distance for every videos
#       for every video calculate distance between it and the query
# decide if the video is similar to any of the query video or whether it is dissimilar
# check the label to see if the video is actually similar
# cumulate mAP score
