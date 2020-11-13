import numpy as np
import pandas as pd
import cv2
import torch
import os
import time
from natsort import natsorted

from Detectors import CNN3D_waveform
from CNN3D_rec import HRCNN
import args

import sys
sys.path.append('..')
#from utils import reader

IN_FPS = 30
IN_IMG_HEIGHT = 1080
IN_IMG_WIDTH = 1920


def main():
    start_time = time.time()

    arg_obj = args.get_input()
    args.print_args(arg_obj)
    number = int(arg_obj.number) - 1
    print('Using number: ', number)

    tk = int(arg_obj.tk)
    sk = int(arg_obj.sk)
    model_load_path = arg_obj.model_load_path
    arg_obj.fps = IN_FPS

    ## Create processing objects and the frame grabber
    shape_predictor = arg_obj.shape_predictor_path

    if model_load_path is None:
        if sk > 1:
            model_load_path = 'model_weights/3dcnn_sk%d' % sk
        else:
            model_load_path = 'model_weights/3dcnn_tk%d' % tk

    ## Make sure a valid model path was given
    if not os.path.exists(model_load_path):
        print('Incorrect path to model weights for 3dcnn. Make sure sk is in \
               [1,20] and tk is in [3,5,7,...,25]. Exiting.')
        return -1

    ## Use GPU if CUDA is configured and load model to correct device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hrcnn = HRCNN(drop_p=0, t_kern=tk).float().to(device)

    ## Load model specified by the path
    checkpoint = torch.load(model_load_path, map_location=device)
    hrcnn.load_state_dict(checkpoint['model_state_dict'])
    hrcnn.eval()

    split='train'
    #split='val'
    #split='test'
    vid_paths, lmrk_paths, out_paths = get_paths(split=split, part_number=number)

    out_dir = '/'.join(out_paths[0].split('/')[:-1])
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    print(vid_paths.shape)
    print(lmrk_paths.shape)
    print(out_paths.shape)
    print()

    for i in range(len(vid_paths)):
        vid_path = vid_paths[i]
        lmrk_path = lmrk_paths[i]
        out_path = out_paths[i]
        lmrks = read_lmrks(lmrk_path)
        all_bad, lmrks = clean_lmrks(lmrks)
        video = prep_video(vid_path, lmrks, all_bad=all_bad)
        waveform = CNN3D_waveform(hrcnn, video, IN_FPS, sk, device=device)
        np.save(out_path, waveform)
    end_time = time.time()
    print('Took %.3f seconds.' % (end_time - start_time))
    return


def get_paths(split, part_number):
    vid_paths = []
    lmrk_paths = []
    out_paths = []
    if split == 'train':
        preproc_dir = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/preprocessed/train'
        data_dir = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/data/train'
        out_dir = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/rPPG/waveforms/train'
        parts = np.array(natsorted(os.listdir(data_dir)))
        part = parts[part_number]
        part_vid_dir = os.path.join(data_dir, part)
        part_lmrk_dir = os.path.join(preproc_dir, part)
        part_out_dir = os.path.join(out_dir, part)
        vids = [p for p in natsorted(os.listdir(part_vid_dir)) if p.endswith('.mp4')]
        vid_paths = [os.path.join(part_vid_dir, p) for p in vids]
        vid_ids = [os.path.splitext(x)[0].split('/')[-1] for x in vids]
        lmrk_paths = [os.path.join(part_lmrk_dir, x+'.csv') for x in vid_ids]
        out_paths = [os.path.join(part_out_dir, x+'.npy') for x in vid_ids]
    else:
        if split == 'val':
            preproc_dir = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/preprocessed/validation'
            data_dir = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/data/validation'
            out_dir = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/rPPG/waveforms/validation'
        elif split == 'test':
            preproc_dir = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/preprocessed/test/processed_test_all_but_aligned_files'
            data_dir = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/data/test'
            out_dir = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/rPPG/waveforms/test'
        else:
            print('Incorrect split passed to function get_paths(split). Exiting...')
            sys.exit(-1)
        vids = [p for p in natsorted(os.listdir(data_dir)) if p.endswith('.mp4')]
        vid_paths = [os.path.join(data_dir, p) for p in vids]
        vid_ids = [os.path.splitext(x)[0].split('/')[-1] for x in vids]
        lmrk_paths = [os.path.join(preproc_dir, x+'.csv') for x in vid_ids]
        out_paths = [os.path.join(out_dir, x+'.npy') for x in vid_ids]
    vid_paths = np.hstack(vid_paths)
    lmrk_paths = np.hstack(lmrk_paths)
    out_paths = np.hstack(out_paths)
    assert(vid_paths.shape == lmrk_paths.shape)
    assert(out_paths.shape == lmrk_paths.shape)
    return vid_paths, lmrk_paths, out_paths



def clean_lmrks(lmrks):
    all_bad = False
    bad_idcs = bad_lmrks(lmrks)
    if len(lmrks) == len(bad_idcs):
        all_bad = True
    clean_lmrks = close_good_lmrks(lmrks, bad_idcs)
    assert(lmrks.shape == clean_lmrks.shape)
    return all_bad, clean_lmrks


def bad_lmrks(lmrks):
    zero_row = np.zeros_like(lmrks[0])
    nan_row = np.full_like(lmrks[0], np.nan)
    zero_idcs = np.where(np.all(lmrks == zero_row, axis=(1,2)))[0]
    nan_idcs = np.where(np.isnan(lmrks).sum((1,2)) > 0)[0]
    idcs = np.sort(np.hstack((zero_idcs, nan_idcs)))
    return idcs


def close_good_lmrks(lmrks, bad_idcs):
    consec_idcs = consecutive(bad_idcs)
    for consec in consec_idcs:
        if consec.size > 0:
            upper_idx = consec[-1] + 1
            lower_idx = consec[0] - 1
            if upper_idx == lmrks.shape[0]:
                upper_idx = lower_idx
            if lower_idx == -1:
                lower_idx = upper_idx
            split_idx = int((consec[-1] + consec[0]) / 2)
            lmrks[consec[:split_idx]] = lmrks[lower_idx]
            lmrks[consec[split_idx:]] = lmrks[upper_idx]
    return lmrks


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def prep_video(vid_path, landmarks, all_bad=False):
    ## Define video reading object
    cap = cv2.VideoCapture(vid_path, cv2.CAP_FFMPEG)
    prepped_video = []
    i = 0
    ## Iterate through the video and crop and resize from face bboxes
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                landmark = landmarks[i]
            except:
                pass
            landmark = landmark.astype(int)

            if not all_bad:
                bbox = get_bbox(landmark)
                bbox = get_square_bbox(bbox)
            else:
                bbox = [0,0,IN_IMG_WIDTH,IN_IMG_HEIGHT]

            crop_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            resized_frame = cv2.resize(crop_frame, (64,64),
                    interpolation=cv2.INTER_CUBIC)
            prepped_video.append(resized_frame)
            i += 1
        else:
            break

    ## Clean up capture object and any open windows
    cap.release()
    cv2.destroyAllWindows()

    ## Video must be in range [0,1] and in order (CxTxHxW)
    prepped_video = np.stack(prepped_video).astype(np.float32)
    prepped_video = prepped_video / 255
    prepped_video = np.transpose(prepped_video, (3, 0, 1, 2))
    return prepped_video


def read_lmrks(lmrk_path):
    df = pd.read_csv(lmrk_path)
    df.columns = df.columns.str.replace(' ', '')
    x_lmrks = [df['x_%d' % i] for i in range(0, 68)]
    y_lmrks = [df['y_%d' % i] for i in range(0, 68)]
    x_lmrks = np.asarray(x_lmrks).T
    y_lmrks = np.asarray(y_lmrks).T
    lmrks = np.stack((x_lmrks, y_lmrks), -1)
    return lmrks


def get_bbox(lmrks):
    x_min, y_min = lmrks.min(axis=0)
    x_max, y_max = lmrks.max(axis=0)
    x_diff = x_max - x_min
    x_upper_pad = x_diff * 0.05
    x_lower_pad = x_diff * 0.05
    x_min -= x_upper_pad
    x_max += x_lower_pad
    if x_min < 0:
        x_min = 0
    if x_max > IN_IMG_WIDTH:
        x_max = IN_IMG_WIDTH
    y_diff = y_max - y_min
    y_upper_pad = y_diff * 0.3
    y_lower_pad = y_diff * 0.05
    y_min -= y_upper_pad
    y_max += y_lower_pad
    if y_min < 0:
        y_min = 0
    if y_max > IN_IMG_HEIGHT:
        y_max = IN_IMG_HEIGHT
    bbox = np.array([x_min, y_min, x_max, y_max]).astype(int)
    return bbox


def shift_inside_frame(x1,y1,x2,y2):
    if y1 < 0:
        y2 -= y1
        y1 -= y1
    if y2 > IN_IMG_HEIGHT:
        shift = y2 - IN_IMG_HEIGHT
        y1 -= shift
        y2 -= shift

    if x1 < 0:
            x2 -= x1
            x1 -= x1
    if x2 > IN_IMG_WIDTH:
        shift = x2 - IN_IMG_WIDTH
        x1 -= shift
        x2 -= shift

    return x1,y1,x2,y2


def get_square_bbox(bbox):
    x1,y1,x2,y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x1,y1,x2,y2 = shift_inside_frame(x1,y1,x2,y2)
    w = x2 - x1
    h = y2 - y1

    ## Push the rectangle out into a square
    if w > h:
        d = w - h
        pad = int(d/2)
        y1 -= pad
        y2 += pad + (d % 2 == 1)
        x1,y1,x2,y2 = shift_inside_frame(x1,y1,x2,y2)
    elif w < h:
        d = h - w
        pad = int(d/2)
        x1 -= pad
        x2 += pad + (d % 2 == 1)
        x1,y1,x2,y2 = shift_inside_frame(x1,y1,x2,y2)

    if x1 < 0:
        x1 = 0
    if x2 > IN_IMG_WIDTH:
        x2 = IN_IMG_WIDTH
    if y1 < 0:
        y1 = 0
    if y2 > IN_IMG_HEIGHT:
        y2 = IN_IMG_HEIGHT

    w = x2 - x1
    h = y2 - y1
    return int(x1), int(y1), int(x2), int(y2)



if __name__ == '__main__':
    main()

