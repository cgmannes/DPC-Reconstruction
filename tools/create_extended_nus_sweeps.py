import copy
import numpy as np
import os
import pickle
import torch

from tqdm.autonotebook import tqdm


if __name__ == '__main__':

    split = 'train'
    data_root = './data/nuscenes/'
    if split=='train':
        infos_path = data_root + f'nuscenes_infos_{split}.pkl'
    elif split=='val':
        infos_path = data_root + f'nuscenes_infos_{split}.pkl'
    elif split=='test':
        infos_path = data_root + f'nuscenes_infos_{split}.pkl'
    else:
        raise ValueError('split must be either train, val, or test')
    

    with open(infos_path, 'rb') as f:
        infos = pickle.load(f)

    prev_sweeps = None
    for idx, info in tqdm(enumerate(infos['infos']), total=len(infos), desc='Extend nuScenes sweeps'):
        if len(info['sweeps'])==0:
            prev_sweeps = None
        elif prev_sweeps is None:
            prev_sweeps = [copy.deepcopy(d) for d in info['sweeps']]
        else:
            info['sweeps'] = prev_sweeps + info['sweeps']
            prev_sweeps = [copy.deepcopy(d) for d in info['sweeps']]
            if len(prev_sweeps)>30:
                prev_sweeps = prev_sweeps[-30:]


    save_path = data_root + f'nuscenes_infos_{split}_40_sweeps.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(infos, f)
