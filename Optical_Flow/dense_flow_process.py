#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import sys
import argparse
from glob import iglob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--number', type=int, default=None, required=False)
parser.add_argument('--split', required=True)
args = parser.parse_args()

video_dir = 'data/'
video_dir = os.path.join(video_dir, args.split)
if args.split == 'train':
    video_dir = os.path.join(video_dir, 'dfdc_train_part_' + str(args.number - 1))

out_dir = 'flow_only/'
out_dir = os.path.join(out_dir, args.split + '_flow_processed')

if not os.path.exists(out_dir):
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass  # race condition, probably!

for video in iglob(os.path.join(video_dir, '*.mp4')):
   vc = cv2.VideoCapture(video)

   fourcc = cv2.VideoWriter_fourcc(*'XVID')

   # Read first frame
   ret, first_frame = vc.read()
   prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

   video_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
   video_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

   # Create mask
   mask = np.zeros_like(first_frame)
   # Sets image saturation to maximum
   mask[..., 1] = 255
   vid_name = video.rsplit('.', 1)[0]

   out_filename = (vid_name + '_output.avi')
   out = cv2.VideoWriter(out_filename, fourcc, 20.0, (video_width, video_height))
   # out = cv2.VideoWriter('video.mp4', -1, 1, (600, 600))

   while vc.isOpened():
      # Read a frame from video
      ret, frame = vc.read()
      if not ret:
         break

      # Convert new frame format`s to gray scale and resize gray frame obtained
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # gray = cv2.resize(gray, None, fx=scale, fy=scale)

      # Calculate dense optical flow by Farneback method
      # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
      flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale=0.5,
                     levels=5, winsize=11, iterations=5,
                     poly_n=5, poly_sigma=1.1, flags=0)
      # Compute the magnitude and angle of the 2D vectors
      magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
      # Set image hue according to the optical flow direction
      mask[..., 0] = angle * 180 / np.pi / 2
      # Set image value according to the optical flow magnitude (normalized)
      mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
      # Convert HSV to RGB (BGR) color representation
      rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

      dense_flow = cv2.addWeighted(frame, 1, rgb, 2, 0)
      out.write(dense_flow)
      # Update previous frame
      prev_gray = gray

   vc.release()

   out.release()

   cv2.destroyAllWindows()
