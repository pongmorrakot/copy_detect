import subprocess
import os

cmd = 'python rmac_features.py inpath /home/ubuntu/Desktop/copy_detect/YFCC100M_pca/ outpath --fps=1 --resnet_level=29 --workers=4 --b_s=32'
vid_path = "/home/ubuntu/Desktop/CC_WEB_Video/video/"
ann_path = "/home/ubuntu/Desktop/CC_WEB_Video/CC_WEB_Video_List.txt"

def get_cmd(inpath, outpath, fps=1, resnet_level=29, workers=4, b_s=32):
    return ['python', 'rmac_features.py', inpath, '/home/ubuntu/Desktop/copy_detect/YFCC100M_pca/', outpath, '--fps=%d'%fps, '--resnet_level=%d'%resnet_level, '--workers=%d'%workers,'--b_s=%d'%b_s]

vid_list = []
for entry in open(ann_path, "r"):
    items = entry.split()
    vid_list.append([items[0], items[1], vid_path + items[1] + "/" + items[3]])

for vid in vid_list:
    outpath = "./pca_features/%s/" % vid[1]
    # print(get_cmd(vid[2], outpath))
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    subprocess.run(get_cmd(vid[2], outpath))
