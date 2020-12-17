import os
import json
import subprocess
import collections

root_path = "/media/ubuntu/Elements/ActivityNet/vid_full/"
out_path = "/media/ubuntu/Elements/ActivityNet/processed/"
label_path = "/media/ubuntu/Elements/ActivityNet/activity_net.v1-3.min.json"
list_path = "/media/ubuntu/Elements/ActivityNet/video_reloc-master/data/train.json"


def cut_vid(vid_name, start_sec, end_sec, fps, out_name):
    # vid_path = root_path + vid_name
    # out_name = out_path + out_name
    if not os.path.exists(out_name):
        os.mkdir(out_name)
    cmd = "ffmpeg -i {vid} -ss {start} -to {end} -r {fps} -async 1 {out_name}frame%04d.jpg"
    cmd = cmd.format(vid=vid_name, start=start_sec, end=end_sec, fps=fps, out_name=out_name)
    os.system(cmd)



def read_json(label_path):
    entry = json.loads(open(label_path, "r").read())
    taxo = entry["taxonomy"]
    label_list = {}
    for i in taxo:
        label_list[i["nodeName"]] = i["nodeId"]
        label_list[i["parentName"]] = i["parentId"]
    # print(label_list)
    data = entry["database"]
    train_list = []
    test_list = []
    val_list = []
    for i in data.items():
        if i[1]["subset"] == "training":
            train_list.append(i)
        elif i[1]["subset"] == "testing":
            test_list.append(i)
        elif i[1]["subset"] == "validation":
            val_list.append(i)
        else:
            print(i)
    return label_list, train_list, test_list, val_list


def gen_train_list(label_list, da_list, out_path):
    final_list = ""
    n = 0
    for i in da_list:
        vid_name = i[0]
        for a in i[1]["annotations"]:
            # print(vid_name + " " + str(a))
            start, end = a["segment"]
            label = a["label"]
            final_list += "t{num:05d} {vid_name} {label} {start} {end}\n".format(num=n, vid_name=vid_name, label=label_list[label],start=start, end=end)
            n +=1
    open(out_path + "_train_list.txt", "w+").write(final_list)

def gen_test_list(label_list, da_list, out_path):
    final_list = ""
    for i in da_list:
        vid_name = i[0]
        final_list += "{vid_name}\n".format(vid_name=vid_name)
    open(out_path + "_test_list.txt", "w+").write(final_list)

def gen_val_list(label_list, da_list, out_path):
    final_list = ""
    n = 0
    for i in da_list:
        vid_name = i[0]
        for a in i[1]["annotations"]:
            # print(vid_name + " " + str(a))
            start, end = a["segment"]
            label = a["label"]
            final_list += "v{num:05d} {vid_name} {label} {start} {end}\n".format(num=n, vid_name=vid_name, label=label_list[label],start=start, end=end)
            n +=1
    open(out_path + "_val_list.txt", "w+").write(final_list)

def create_annotation():
    # vid_list = make_list(list_path)
    label_list, train_list, test_list, val_list = read_json(label_path)
    # print(test_list)
    gen_train_list(label_list, train_list, out_path)
    gen_test_list(label_list, test_list, out_path)
    gen_val_list(label_list, val_list, out_path)

    new_list = []

    for k,v  in list(label_list.items()):
       if v is None:
          del label_list[k]

    for i in range(400):
        new_list.append("")
    for e in label_list.items():
        # print(e)
        new_list[int(e[1])] = e[0]
    label = ""
    for i, v in enumerate(new_list):
        label += "{i} {v}\n".format(i=i,v=v)
    open(out_path + "_classLabel.txt", "w+").write(label)
    # print(new_list)
    # print(train_list)


def extract_vid(list_path, vid_path, out_path):
    entries = open(list_path,"r").readlines()
    for e in entries:
        out_name, vid_name, label, start, end = e.split()
        cut_vid(vid_path + "v_" + vid_name + ".mp4", start, end, 8, out_path+out_name+"/")
        print(vid_name + "\t extracted")


def extract_all():
    extract_vid(out_path+"_train_list.txt", root_path+"train_val/", out_path)
    extract_vid(out_path+"_val_list.txt", root_path+"train_val/", out_path)

extract_all()
