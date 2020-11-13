#!/bin/bash

#$ -M jspeth@nd.edu
#$ -o /scratch365/jspeth/log
#$ -e /scratch365/jspeth/log
#$ -m abe
#$ -q long
#$ -pe smp 12
#$ -t 1-8:1
#$ -N openface_deepfakes

conda activate /afs/crc.nd.edu/user/j/jspeth/.conda/envs/BobTorch/
SINGULARITYENV_PATH=/bin:/usr/bin 

python singularity_submit.py ${SGE_TASK_ID}

