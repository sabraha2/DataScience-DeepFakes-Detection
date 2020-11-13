#####
## Licensing for NumPy is provided in licenses/NumPy_License.txt
#####

import cv2
import numpy as np
import pandas as pd
import torch
import math
import shutil
import os
from copy import deepcopy


def CNN3D_waveform(hrcnn, prepped_video, fps, sk, device='cpu'):
    '''
    Returns a waveform from the video after preprocessing and feeding volumes
    to a 3DCNN model.

    Args:
        input_file (str): path to video file.
        frame_processor (obj): processor for face detection and averaging.
        hrcnn (obj): Pre-trained PyTorch model for predicting pulse over 135
                     frame segments.
        video_meta (pymediainfo obj): metadata from the video
        n (int): Runs the face detector every n frames.
        device (str): cuda:X for using GPU number X, but defaults to cpu.
        show (bool): Whether to show the landmarking and bboxes when processing

    Returns:
        waveform (np.array): 1D temporal waveform of the estimated pulse
    '''

    vlen = prepped_video.shape[1]

    #fps_model = 90/sk
    seglen = 1.5*fps
    if seglen % 2 != 0:
        seglen += (2 - (seglen % 2))
    seglen = int(seglen)
    step = int(seglen/2)
    print(vlen, fps, sk, seglen, step)

    ## Much faster to load batches to GPU and process all of them at once
    batch_size = 8
    batch_segs = np.zeros((batch_size, 3, seglen, 64, 64))
    tot_iters = int(vlen/batch_size)
    rem_iters = vlen % batch_size
    seg_preds = []

    ## Run through full batches of largest size possible
    seg_start_idcs = np.arange(0, vlen-seglen+1, step)
    num_batches = len(seg_start_idcs)/batch_size
    for batch_idx in range(0, int(num_batches)):
        st_idx = batch_idx*batch_size
        end_idx = st_idx + batch_size
        seg_starts = seg_start_idcs[st_idx:end_idx]
        for i,seg_start in enumerate(seg_starts):
            batch_segs[i] = prepped_video[:, seg_start:seg_start+seglen]
        with torch.set_grad_enabled(False):
            outputs = hrcnn(torch.from_numpy(batch_segs).float().to(device))
            output_copy = deepcopy(outputs.cpu().numpy())
            seg_preds.append(output_copy)
            del outputs

    num_segs_remain = len(seg_start_idcs) - int(num_batches)*batch_size
    seg_start_idcs_remain = seg_start_idcs[-num_segs_remain:]
    batch_segs = batch_segs[:len(seg_start_idcs_remain)]
    for i,seg_start in enumerate(seg_start_idcs_remain):
        batch_segs[i] = prepped_video[:, seg_start:seg_start+seglen]
    with torch.set_grad_enabled(False):
        outputs = hrcnn(torch.from_numpy(batch_segs).float().to(device))
        output_copy = deepcopy(outputs.cpu().numpy())
        seg_preds.append(output_copy)
        del outputs

    seg_preds = np.vstack(seg_preds)
    print('Seg_preds: ', seg_preds.shape)

    ## Overlap-adding for a cleaner output
    waveform = np.zeros(vlen)
    print('Waveform: ', waveform.shape)
    hanning_window = np.hanning(seglen)
    for j, seg_pred in enumerate(seg_preds):
        start = int(j*step)
        end = start+seglen
        print(start, end)
        if end < vlen:
            waveform[start:end] += (hanning_window * ((seg_pred - np.mean(seg_pred)) / np.std(seg_pred)))
    waveform = (waveform - np.mean(waveform)) / np.std(waveform)

    return waveform


