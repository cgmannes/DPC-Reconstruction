import copy
import numpy as np
import os
import pickle
import torch

from tqdm.autonotebook import tqdm


if __name__ == '__main__':

    sweeps_num = 20
    split = 'val'
    data_root = './data/waymo/kitti_format/'
    if split=='train':
        infos_path = data_root + f'waymo_infos_{split}.pkl'
    elif split=='val':
        infos_path = data_root + f'waymo_infos_{split}.pkl'
    else:
        raise ValueError('split must be either train or val')
    

    with open(infos_path, 'rb') as f:
        infos = pickle.load(f)

    prev_sweeps = []
    for idx, info in tqdm(enumerate(infos), total=len(infos), desc='Extend Waymo sweeps'):
        
        if len(info['sweeps'])==0:
            prev_sweeps = []
        elif 0<=len(prev_sweeps)<sweeps_num:
            prev_sweeps.insert(0,info['sweeps'][0])
            info['sweeps'] = prev_sweeps
        else:
            prev_sweeps.insert(0,info['sweeps'][0])
            info['sweeps'] = prev_sweeps[:-1]


    save_path = data_root + f'waymo_infos_{split}_{sweeps_num}_sweeps.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(infos, f)
