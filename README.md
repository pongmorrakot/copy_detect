# copy_detect

i3d.py : contains the i3d model

i3dBERT.py : contains the i3d+BERT model

i3d_features.py : script to extract feature using i3d or i3d+Bert

Command: python i3d_features.py {weight_path} {in_path} {out_path}
Note: out_path should include the filename e.g. /home/ubuntu/feature.pk

==============================================================================================

extract_frame.py : function to extract frame from a video; make use of ffmpeg

compute.py : functions to compute similarity between video features

evaluate.py : script for evaluate near duplicate video retrieval(NDVR) performance

TODO: complete the documentation
