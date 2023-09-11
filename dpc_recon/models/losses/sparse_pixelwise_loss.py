# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmgen.models.builder import MODULES
from mmgen.models.losses.pixelwise_loss import *
from mmgen.models.losses.pixelwise_loss import l1_loss as mmgen_l1_loss


@MODULES.register_module()
class SparseL1Loss(L1Loss):
    """Modified L1 loss weight for a sparse feature map.

    Args:
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (float | None, optional): Average factor when computing the
            mean of losses. Defaults to ``None``.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary. If ``None``, this module will
            directly pass the input data to the loss function.
            Defaults to None.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_l1'.
    """
    def __init__(self, beta, gamma, nonempty_weight, height, width, pixel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.gamma = gamma
        self.nonempty_weight = nonempty_weight
        self.height = height
        self.width = width
        self.pixel_size = pixel_size
        self.dist_weight = self.logistic_decay_weight(self.height, self.width, self.pixel_size).cuda()

    def forward(self, *args, **kwargs):
        """Forward function.
        If ``self.data_info`` is not ``None``, a dictionary containing all of
        the data and necessary modules should be passed into this function.
        If this dictionary is given as a non-keyword argument, it should be
        offered as the first argument. If you are using keyword argument,
        please name it as `outputs_dict`.
        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function, ``l1_loss``.
        """
        # use data_info to build computational path
        if self.data_info is not None:
            # parse the args and kwargs
            if len(args) == 1:
                assert isinstance(args[0], dict), (
                    'You should offer a dictionary containing network outputs '
                    'for building up computational graph of this loss module.')
                outputs_dict = args[0]
            elif 'outputs_dict' in kwargs:
                assert len(args) == 0, (
                    'If the outputs dict is given in keyworded arguments, no'
                    ' further non-keyworded arguments should be offered.')
                outputs_dict = kwargs.pop('outputs_dict')
            else:
                raise NotImplementedError(
                    'Cannot parsing your arguments passed to this loss module.'
                    ' Please check the usage of this module')
            # link the outputs with loss input args according to self.data_info
            loss_input_dict = {
                k: outputs_dict[v]
                for k, v in self.data_info.items()
            }
            kwargs.update(loss_input_dict)
            kwargs.update(
                dict(weight=self.loss_weight, reduction=self.reduction))
            gt_bboxes_3d = kwargs.pop('gt_bboxes_3d')
            gt_labels_3d = kwargs.pop('gt_labels_3d')
            get_l1_targets_func = kwargs.pop('get_l1_targets_func')
            target_mask = kwargs['target'] != 0
            N = target_mask[target_mask==True].shape[0]
            N_tilde = target_mask[target_mask==False].shape[0]
            # target_size = np.prod(list(target_mask.shape))
            loss = mmgen_l1_loss(**kwargs)
            heatmap_masks = get_l1_targets_func(
                gt_bboxes_3d, gt_labels_3d)
            heatmap_masks = torch.cat(heatmap_masks, dim=0).cuda()
            beta_loss = self.beta * (loss[target_mask].sum() / N)
            gamma_loss = self.gamma * (loss[~target_mask].sum() / N_tilde)
            l1_loss = beta_loss + gamma_loss
            # loss = loss * self.dist_weight * heatmap_masks
            # nonempty_loss = self.nonempty_weight * (loss[target_mask].sum() / target_size)
            # empty_loss = (1 - self.nonempty_weight) * (loss[~target_mask].sum() / target_size)
            # l1_loss = nonempty_loss + empty_loss
            return l1_loss, heatmap_masks
        else:
            # if you have not define how to build computational graph, this
            # module will just directly return the loss as usual.
            gt_bboxes_3d = kwargs.pop('gt_bboxes_3d')
            gt_labels_3d = kwargs.pop('gt_labels_3d')
            get_l1_targets_func = kwargs.pop('get_l1_targets_func')
            target_mask = kwargs['target'] != 0
            N = target_mask[target_mask==True].shape[0]
            N_tilde = target_mask[target_mask==False].shape[0]
            target_size = np.prod(list(target_mask.shape))
            loss = mmgen_l1_loss(
                *args,
                weight=self.loss_weight,
                reduction=self.reduction,
                avg_factor=self.avg_factor,
                **kwargs)
            heatmap_masks = get_l1_targets_func(
                gt_bboxes_3d, gt_labels_3d)
            heatmap_masks = torch.cat(heatmap_masks, dim=0)
            beta_loss = self.beta * (loss[target_mask].sum() / N)
            gamma_loss = self.gamma * (loss[~target_mask].sum() / N_tilde)
            l1_loss = beta_loss + gamma_loss
            # loss = loss * self.dist_weight * heatmap_masks
            # nonempty_loss = self.nonempty_weight * (loss[target_mask].sum() / target_size)
            # empty_loss = (1 - self.nonempty_weight) * (loss[~target_mask].sum() / target_size)
            # l1_loss = nonempty_loss + empty_loss
            return l1_loss, heatmap_masks

    def logistic_decay_weight(self, height, width, pixel_size):
        '''
        Function for generating a weight tensor based on the Euclidean
        distance from the ego vehicle. The implemented function is
        modified such that the denominator of the ratio raised to the
        power of 2.8 is altered from 25 to 34.64. This was chosen such
        that the furthest pixel are weighted by approx. 0.1.

        Args:
            height (int): Height of the loss tensor.
            width (int): Width of the loss tensor.
            pixel_size (float): Distance in meters covered by each pixel
                                in the loss tensor.

        Based on work from:
            https://www.researchgate.net/figure/a-Comparison-of-the-three-different-decay-functions-Gaussian-logistic-and_fig1_305074253
            
            f(d) = 1 / (1 + (d/25)**2.8)
        '''
        import matplotlib.pyplot as plt

        grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))

        y_steps = (grid_y * pixel_size) + (pixel_size / 2)
        x_steps = (grid_x * pixel_size) + (pixel_size / 2)

        y_coords = -1*(y_steps - (height / 2 * pixel_size))
        x_coords = (x_steps - (width / 2 * pixel_size))

        euc_dist = (y_coords**2 + x_coords**2)**(1/2)
        dist_weight = 1 / (1 + (euc_dist / 34.64)**2.8)

        # plt.figure(figsize=(15,15))
        # ax = plt.subplot(1,1,1)
        # ax.set_title(f'Heatmap for Logistic Decay of Euclidean Distance Weight Tensor',
        #              size=20, fontweight='bold')
        # # Plot the heatmap.
        # im = ax.imshow(dist_weight)
        # # Create colorbar.
        # ax.figure.colorbar(im, ax=ax)

        # plt.tight_layout()
        # plt.savefig(f'./work_dirs/domain_adaptation_experiments/logistic_decay_weight_tensor.png')
        # plt.close()

        return dist_weight.unsqueeze(0).unsqueeze(0)