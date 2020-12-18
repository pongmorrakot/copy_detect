# copy_detect

extract_ActNet.py : functions to extract clip of videos from activityNet

train_i3dBERT : code for traning I3D+BERT model on ActivityNet

i3d.py : contains the i3d model

i3dBERT.py : contains the i3d+BERT model

i3d_features.py : script to extract feature using i3d or i3d+Bert

Command: python i3d_features.py {weight_path} {in_path} {out_path}
Note: out_path should include the filename e.g. /home/ubuntu/feature.pk

============================================================================================

rmac_features.py : script to extract features using RMAC; got from facebookresearch

run_rmac.py : script to automate rmac_features.py

============================================================================================

train_visil.py : functions to train ViSiL on CC_WEB_Video
 
visil.py : Implementation of ViSiL network

============================================================================================

compute.py : functions to compute similarity between video features

evaluate.py : script for evaluate near duplicate video retrieval(NDVR) performance

extract_frame.py : function to extract frame from a video; make use of ffmpeg

feature_extract.py : extract feature using pretrained i3d models

utils.py : utilities functions

vgg.py : Vgg model(incomplete)


Code I got from other places (use mainly for reference purposes):

rgb_I3D.py

frame_extract.py
