import os
import sys
import glob
import shutil
import codecs
from tqdm import tqdm_notebook as tqdm

import pandas as pd
import numpy as np
import time
from multiprocessing import Pool

from PIL import Image

PATH = './positives/'

# Extract Frame
class FrameExtractor:
    # key uniform scene
    def __init__(self, path=PATH):
        self.train_path = path + 'train/'
        self.test_path = path + 'test/'
        self.train_query_path = self.train_path + 'query/'
        self.refer_path = self.train_path + 'refer/'
        #self.test_query_path = self.test_path + 'query/'
        self.test_query_path = self.test_path + 'query/'
        #self.train_df = pd.read_csv(self.train_path + 'train.csv')
        self.train_query_paths = self._get_videos(self.train_query_path)
        self.test_query_paths = self._get_videos(self.test_query_path)
        self.refer_paths = self._get_videos(self.refer_path)

    def _get_videos(self, path):
        print(path)
        video_paths = glob.glob(path + '*.*')
        print(video_paths)
        return video_paths

    def extract_keyframe(self, video_path, frame_path):
        video_id = video_path.split('/')[-1][:-4]

        if not os.path.exists(frame_path):
            os.mkdir(frame_path)
        if not os.path.exists(frame_path + video_id):
            os.mkdir(frame_path + video_id)

        # 抽取关键帧（I帧）
        command = ['ffmpeg', '-i', video_path,
                   '-vf', '"select=eq(pict_type\,I)"',
                   '-vsync', 'vfr', '-qscale:v', '2',
                   '-f', 'image2',
                   frame_path + '{0}/{0}_%05d.jpg'.format(video_id)]
        os.system(' '.join(command))

        # 抽取视频关键帧时间
        command = ['ffprobe', '-i', video_path,
                   '-v', 'quiet', '-select_streams',
                   'v', '-show_entries', 'frame=pkt_pts_time,pict_type|grep',
                   '-B', '1', 'pict_type=I|grep pkt_pts_time', '>',
                   frame_path + '{0}/{0}.log'.format(video_id)]
        os.system(' '.join(command))

    def _extract_keyframe(self, param):
        self.extract_keyframe(param[0], param[1])

    def extract_uniformframe(self, video_path, frame_path, frame_per_sec=1):
        video_id = video_path.split('/')[-1][:-4]
        print(frame_path + video_id)

        if not os.path.exists(frame_path):
            os.mkdir(frame_path)
        if not os.path.exists(frame_path + video_id):
            os.mkdir(frame_path + video_id)

        # -r 指定抽取的帧率，即从视频中每秒钟抽取图片的数量。1代表每秒抽取一帧。
        command = ['ffmpeg', '-i', video_path,
                   '-r', str(frame_per_sec),
                   '-q:v', '2', '-f', 'image2',
                   frame_path + '{0}/{0}_%08d.000000.jpg'.format(video_id)]


        print(command)
        os.system(' '.join(command))


    def _extract_uniformframe(self, param):
        self.extract_uniformframe(param[0], param[1], param[2])

    # 关键帧用时间戳重命名
    def _rename(self, video_paths, frame_path, mode='key', frame_per_sec=1):
        for path in video_paths[:]:
            video_id = path.split('/')[-1][:-4]
            id_files = glob.glob(frame_path + video_id + '/*.jpg')
            # IMPORTANT!!!
            id_files.sort()
            if mode == 'key':
                id_times = codecs.open(frame_path + '{0}/{0}.log'.format(video_id)).readlines()
                id_times = [x.strip().split('=')[1] for x in id_times]

                for id_file, id_time in zip(id_files, id_times):
                    shutil.move(id_file, id_file[:-9] + id_time.zfill(15) + '.jpg')
            else:
                id_time = 0.0
                for id_file in id_files:
                    shutil.move(id_file, id_file[:-19] + '{:0>15.4f}'.format(id_time) + '.jpg')
                    id_time += 1.0 / frame_per_sec

    def extract(self, mode='key', num_worker=5, frame_per_sec_q=1, frame_per_sec_r=1):
        if mode == 'key':
            pool = Pool(processes=num_worker)
            for path in self.train_query_paths:
                pool.apply_async(self._extract_keyframe, ((path, self.train_path + 'query_keyframe/'),))

            for path in self.test_query_paths:
                # pool.apply_async(self._extract_keyframe, ((path, self.test_path + 'query_keyframe/'),))
                pool.apply_async(self._extract_keyframe, ((path, self.test_path + 'query_keyframe/'),))

            for path in self.refer_paths:
                pool.apply_async(self._extract_keyframe, ((path, self.train_path + 'refer_keyframe/'),))

            pool.close()
            pool.join()

            self._rename(self.train_query_paths, self.train_path + 'query_keyframe/')
            # self._rename(self.test_query_paths, self.test_path + 'query_keyframe/')
            self._rename(self.test_query_paths, self.test_path + 'query_keyframe/')
            self._rename(self.refer_paths, self.train_path + 'refer_keyframe/')

        elif mode == 'uniform':
            print("does this run?")
            pool = Pool(processes=num_worker)
            #for path in self.train_query_paths:
            #    pool.apply_async(self._extract_uniformframe, ((path, self.train_path + 'query_uniformframe/', frame_per_sec_q),))

            for path in self.test_query_paths:
                print("how about this?")
                pool.apply_async(self._extract_uniformframe, ((path, self.test_path + 'query_uniformframe/', frame_per_sec_q),))
                #pool.apply_async(self._extract_uniformframe, ((path, self.test_path + 'query2_uniformframe/', frame_per_sec_q),))

            #for path in self.refer_paths:
            #    pool.apply_async(self._extract_uniformframe, ((path, self.train_path + 'refer_uniformframe/', frame_per_sec_r),))

            pool.close()
            pool.join()

            #self._rename(self.train_query_paths, self.train_path + 'query_uniformframe/',
            #             mode='uniform', frame_per_sec=frame_per_sec_q)
            self._rename(self.test_query_paths, self.test_path + 'query_uniformframe/',
                         mode='uniform', frame_per_sec=frame_per_sec_q)
            #self._rename(self.test_query_paths, self.test_path + 'query_uniformframe/',
            #             mode='uniform', frame_per_sec=frame_per_sec_q)
            #self._rename(self.refer_paths, self.train_path + 'refer_uniformframe/',
            #             mode='uniform', frame_per_sec=frame_per_sec_r)
        else:
            None


frame_extractor = FrameExtractor(PATH)
frame_extractor.extract(mode='uniform', num_worker=1, frame_per_sec_q=1, frame_per_sec_r=1)
