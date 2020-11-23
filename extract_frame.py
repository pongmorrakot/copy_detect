import os

inpath = "/home/ubuntu/Desktop/VCDB/test_pos/"
outpath = "./positive_frames/"
# fps = 16


def extract(inpath, outpath, fps):
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    cmd = "ffmpeg -loglevel quiet -stats -i " + inpath + " -r " + str(fps) + " " + outpath + "frame%04d.jpg"
    # print(cmd)
    os.system(cmd)


# folders = os.listdir(inpath)
# for f in folders:
#     f_path = inpath + f + "/"
#     clips = os.listdir(f_path)
#     os.mkdir(outpath + f)
#     extract(f_path + "/clip1.mp4", outpath + f + "/clip1/", fps)
#     extract(f_path + "/clip2.mp4", outpath + f + "/clip2/", fps)
