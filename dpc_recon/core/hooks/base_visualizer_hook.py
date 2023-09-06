# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import torch
from datetime import date

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

import mmcv
from mmcv.runner import Hook
from mmcv.runner.dist_utils import master_only


class BaseVisualizerHook(Hook):
    """Base visualization hook for mmdet3d_mmgen framework.

    Args:
        output_dir (str): The file path to store visualizations.
        exp_name (str): The name of the experiment being investigated.
        res_name_list (str): The list contains the domain names of results in
            outputs dict. The results in outputs dict must be a torch.Tensor
            with shape (n, c, h, w).
        interval (int): The interval of calling this hook. If set to -1,
            the visualization hook will not be called. Default: -1.
        filename_tmpl (str): Format string used to save images. The output file
            name will be formatted as this args. Default: 'iter_{}.png'.
    """

    def __init__(self,
                 output_dir,
                 exp_name,
                 res_name_list=[],
                 interval=-1,
                 filename_tmpl='iter_{}'):
        self.today = date.today()
        self.today_formatted = self.today.strftime("%Y_%m_%d")
        self.date_exp = f'{self.today_formatted}_{exp_name}'
        self.output_dir = osp.join(output_dir, self.date_exp, 'training_analysis')
        self.res_name_list = res_name_list
        self.interval = interval
        self.filename_tmpl = filename_tmpl

        # Plotting keyword values.
        self.interp='None'
        self.ext=[-54,54,-54,54]
        self.orig='lower'

    @master_only
    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (object): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        
        if self.paired_translation:
            self.paired_translation_visualizations(runner)
        elif not self.paired_translation:
            self.unpaired_translation_visualizations(runner)
        else:
            ValueError(f'The member variable self.paired_translation must be set.')

    @master_only
    def dump_vis_data(self, data_dict, vis_data_dir, filename):
        '''Function for dumping the visualization data of the domain adaptation
        GAN CenterPoint model to a pickle object.
        '''
        output_dir = osp.join(self.output_dir, vis_data_dir)
        mmcv.mkdir_or_exist(output_dir)
        output_dir = osp.join(output_dir, filename)

        for key, val in data_dict.items():
            val = val.numpy() if torch.is_tensor(val) else val
            with open(f'{output_dir}_{key}.pkl', 'wb') as write_file:
                pickle.dump(val, write_file)

    @master_only
    def plot_bboxes(self, bboxes_3d, box_color='orange'):
        '''Function for plotting the bboxes.
        '''
        for box in bboxes_3d.tensor.numpy():
            plt.gca().add_patch(Rectangle(
                box[:2]-box[3:5]/2, box[3], box[4],
                lw=1, ec=f'tab:{box_color}', fc=f'tab:{box_color}', alpha=0.25,
                transform=Affine2D().rotate_around(box[0], box[1], -box[6]) + 
                    plt.gca().transData
            ))
