import numpy as np
from natsort import natsorted
import subprocess
import os
import sys

#### SUBMITTING MANY DIFFERENT PARTS SPLIT INTO QUARTERS 
number = int(sys.argv[1]) - 1
part_number = (number // 4) + 4
quarter = number % 4
vid_root = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/data/train/dfdc_train_part_%d' % part_number
out_root = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/preprocessed/train/dfdc_train_part_%d' % part_number
videos = np.array(natsorted(os.listdir(vid_root)))
videos = np.array_split(videos, 4)[quarter]

#### SUBMITTING VAL
#vid_root = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/data/validaition'
#out_root = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/preprocessed/validaition'
#videos = np.array(natsorted(os.listdir(vid_root)))

#### SUBMITTING TEST
#vid_root = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/data/test'
#out_root = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/preprocessed/test'
#videos = np.array(natsorted(os.listdir(vid_root)))

for video in videos:
    video_path = os.path.join(vid_root, video)
    out_file = video[:-4]
    comm = f"singularity exec --cleanenv --bind /afs/crc.nd.edu/user/j/jspeth/OpenFace/build/bin/classifiers:/home/openface-build/build/bin/classifiers --bind /afs/crc.nd.edu/user/j/jspeth/OpenFace/build/bin/model:/home/openface-build/build/bin/model --bind {vid_root}:/input --bind {out_root}:/output /scratch365/jspeth/openface_latest.sif /home/openface-build/build/bin/FeatureExtraction -f /input/{video} -out_dir /output -of {out_file} -2Dfp -3Dfp -gaze -pdmparams -pose -aus -hogalign -simalign -format_aligned png -format_vis_image png -nomask"

    print(comm)
    subprocess.call(comm.split(' '))


