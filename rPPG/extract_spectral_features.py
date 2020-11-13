import numpy as np
from scipy import signal
import pandas as pd
import os
import time
from natsort import natsorted

import args

import sys
sys.path.append('..')

def main():
    start_time = time.time()

    arg_obj = args.get_input()
    args.print_args(arg_obj)
    number = int(arg_obj.number) - 1

    #split='train'
    split='validation'
    #split='test'
    vid_paths, lmrk_paths, pulse_paths = get_paths(split=split, part_number=number)

    if split == 'train':
        spect_idx = -4
    else:
        spect_idx = -3

    print(pulse_paths.shape)
    print()

    for i in range(len(pulse_paths)):
        pulse_path = pulse_paths[i]
        splits = pulse_path.split('/')
        splits[spect_idx] = 'spectrums'
        spectrum_dir = '/'.join(splits[:-1])
        spectrum_file = splits[-1]
        if not os.path.isdir(spectrum_dir):
            os.makedirs(spectrum_dir)
        spectrum_path = os.path.join(spectrum_dir, spectrum_file)
        print(pulse_path)
        print(spectrum_path)
        pulse = np.load(pulse_path)
        freq, density = spectral_features(pulse)
        print(density.shape)
        np.save(spectrum_path, density)
        print()
    end_time = time.time()
    print('Took %.3f seconds.' % (end_time - start_time))
    return


def spectral_features(sig):
    nperseg = len(sig) // 2
    overlap = nperseg * 0.5
    fftsize = 512
    freq, density = signal.welch(sig, fs=30, nperseg=nperseg, noverlap=overlap, nfft=512, scaling='density')
    return freq, density



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
        if split[:3] == 'val':
            preproc_dir = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/preprocessed/validation'
            data_dir = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/data/validation'
            out_dir = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/rPPG/waveforms/validation'
        elif split == 'test':
            preproc_dir = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/preprocessed/test/processed_crc'
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


if __name__ == '__main__':
    main()

