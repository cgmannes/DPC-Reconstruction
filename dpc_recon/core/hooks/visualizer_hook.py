# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import os.path as osp
import pandas as pd
import scipy
import seaborn as sns
import torch
from datetime import date

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

import mmcv
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only


@HOOKS.register_module('MMDet3dMMGenVisualizationHook')
class VisualizerHook(Hook):
    """Visualization hook for mmdet3d_mmgen framework.

    In this hook, we use the official api `save_image` in torchvision to save
    the visualization results.

    Args:
        output_dir (str): The file path to store visualizations.
        res_name_list (str): The list contains the domain names of results in
            outputs dict. The results in outputs dict must be a torch.Tensor
            with shape (n, c, h, w).
        interval (int): The interval of calling this hook. If set to -1,
            the visualization hook will not be called. Default: -1.
        filename_tmpl (str): Format string used to save images. The output file
            name will be formatted as this args. Default: 'iter_{}.png'.
        rerange (bool): Whether to rerange the output value from [-1, 1] to
            [0, 1]. We highly recommend users should preprocess the
            visualization results on their own. Here, we just provide a simple
            interface. Default: True.
        bgr2rgb (bool): Whether to reformat the channel dimension from BGR to
            RGB. The final image we will save is following RGB style.
            Default: True.
        nrow (int): The number of samples in a row. Default: 1.
        padding (int): The number of padding pixels between each samples.
            Default: 4.
    """

    def __init__(self,
                 output_dir,
                 exp_name,
                 res_name_list=[],
                 interval=-1,
                 filename_tmpl='iter_{}',
                 feat2feat=False,
                 cycle_feat_gan=False):
        self.today = date.today()
        self.today_formatted = self.today.strftime("%Y_%m_%d")
        self.date_data = f'{self.today_formatted}_{exp_name}'
        self.output_dir = osp.join(output_dir, self.date_data, 'training_analysis')
        self.res_name_list = res_name_list
        self.interval = interval
        self.filename_tmpl = filename_tmpl
        self.feat2feat = feat2feat
        self.cycle_feat_gan = cycle_feat_gan

    @master_only
    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (object): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        
        if self.feat2feat:
            sparse_points = runner.data_batch['points']
            agg_points = runner.data_batch['agg_points']
            real_sparse_feature_map = runner.outputs[f'feat2feat_results']\
                                                    [f'real_{self.res_name_list[0]}']
            fake_agg_feature_map = runner.outputs[f'feat2feat_results']\
                                                 [f'fake_{self.res_name_list[1]}']
            real_agg_feature_map = runner.outputs[f'feat2feat_results']\
                                                 [f'real_{self.res_name_list[1]}']
            full_real_sparse_feature_map = runner.outputs[f'feat2feat_results']\
                                                         [f'full_real_{self.res_name_list[0]}']
            full_fake_agg_feature_map = runner.outputs[f'feat2feat_results']\
                                                      [f'full_fake_{self.res_name_list[1]}']
            full_real_agg_feature_map = runner.outputs[f'feat2feat_results']\
                                                      [f'full_real_{self.res_name_list[1]}']
            fft_fake_agg_feature_map = runner.outputs[f'feat2feat_results']\
                                                     [f'fft_fake_{self.res_name_list[1]}']
            fft_real_agg_feature_map = runner.outputs[f'feat2feat_results']\
                                                     [f'fft_real_{self.res_name_list[1]}']
            heatmap_masks = runner.outputs[f'feat2feat_results']\
                                          [f'heatmap_masks']
            gt_bboxes_3d = runner.outputs[f'feat2feat_results']\
                                         [f'gt_bboxes_3d']

            filename = self.filename_tmpl.format(runner.iter + 1)
            save_filename = osp.join(self.output_dir, filename)
            mmcv.mkdir_or_exist(self.output_dir)

            self.save_visualization(sparse_points.data[0][0], agg_points.data,
                                    real_sparse_feature_map, fake_agg_feature_map,
                                    real_agg_feature_map, full_real_sparse_feature_map,
                                    full_fake_agg_feature_map, full_real_agg_feature_map,
                                    fft_fake_agg_feature_map, fft_real_agg_feature_map,
                                    heatmap_masks, gt_bboxes_3d, save_filename, 'Feat2Feat GAN')
        
        if self.cycle_feat_gan:
            sparse_points = runner.data_batch['points']
            agg_points = runner.data_batch['agg_points']
            real_sparse_feature_map = runner.outputs[f'cycle_feat_gan_results']\
                                                    [f'real_{self.res_name_list[0]}']
            fake_agg_feature_map = runner.outputs[f'cycle_feat_gan_results']\
                                                 [f'fake_{self.res_name_list[1]}']
            cycle_feature_map = runner.outputs[f'cycle_feat_gan_results']\
                                              [f'cycle_{self.res_name_list[0]}']
            gan_dist_map = runner.outputs[f'cycle_feat_gan_results']\
                                         [f'gan_dist_map']
            gt_bboxes_3d = runner.outputs[f'cycle_feat_gan_results']\
                                         [f'gt_bboxes_3d']

            filename = self.filename_tmpl.format(runner.iter + 1)
            save_filename = osp.join(self.output_dir, filename)
            mmcv.mkdir_or_exist(self.output_dir)

            self.save_visualization(sparse_points.data[0][0], agg_points.data,
                                    real_sparse_feature_map, fake_agg_feature_map,
                                    cycle_feature_map, None, gan_dist_map,
                                    gt_bboxes_3d, save_filename, 'Cycle Feat GAN')

    @master_only
    def save_visualization(self,
                           sparse_points,
                           agg_points,
                           real_sparse_feature_map,
                           fake_agg_feature_map,
                           real_agg_feature_map,
                           full_real_sparse_feature_map,
                           full_fake_agg_feature_map,
                           full_real_agg_feature_map,
                           fft_fake_agg_feature_map,
                           fft_real_agg_feature_map,
                           heatmap_masks,
                           gt_bboxes_3d,
                           filename,
                           model_type):
        ##################################################################################
        # ANALYZE/DEBUG: Plot and compare the point clouds and feature maps at all stages.
        ##################################################################################
        # Plotting keyword values.
        interp='None'
        ext=[-54,54,-54,54]
        orig='lower'

        feature_map_max = torch.max(real_agg_feature_map)
        feature_map_min = torch.min(real_agg_feature_map)

        plt.figure(figsize=(15*5,15))
        ax1 = plt.subplot(1,5,1)
        ax1.set_title(f'Sparse Point Cloud', size=20, fontweight='bold')
        plt.scatter(
            sparse_points[:,0], sparse_points[:,1],
            s=0.001, alpha=0.5
        )
        for box in gt_bboxes_3d.tensor.numpy():
            plt.gca().add_patch(Rectangle(
                box[:2]-box[3:5]/2, box[3], box[4],
                lw=1, ec='tab:orange', fc='tab:orange', alpha=0.25,
                transform=Affine2D().rotate_around(box[0], box[1], -box[6]) + 
                    plt.gca().transData
            ))
        plt.xlim(-54,54)
        plt.ylim(-54,54)

        ax2 = plt.subplot(1,5,2)
        ax2.set_title(f'Sparse Point Cloud Encoded Feature Map - Real Source Image',
                      size=20, fontweight='bold')
        plt.imshow(real_sparse_feature_map.numpy(), interpolation=interp, extent=ext,
                   origin=orig, vmax=feature_map_max, vmin=feature_map_min)
        plt.colorbar()
        for box in gt_bboxes_3d.tensor.numpy():
            plt.gca().add_patch(Rectangle(
                box[:2]-box[3:5]/2, box[3], box[4],
                lw=1, ec='tab:orange', fc='tab:orange', alpha=0.25,
                transform=Affine2D().rotate_around(box[0], box[1], -box[6]) + 
                    plt.gca().transData
            ))

        ax3 = plt.subplot(1,5,3)
        ax3.set_title(f'{model_type} Feature Map - Fake Target Image', size=20, fontweight='bold')
        plt.imshow(fake_agg_feature_map.numpy(), interpolation=interp, extent=ext,
                   origin=orig, vmax=feature_map_max, vmin=feature_map_min)
        plt.colorbar()
        for box in gt_bboxes_3d.tensor.numpy():
            plt.gca().add_patch(Rectangle(
                box[:2]-box[3:5]/2, box[3], box[4],
                lw=1, ec='tab:orange', fc='tab:orange', alpha=0.25,
                transform=Affine2D().rotate_around(box[0], box[1], -box[6]) + 
                    plt.gca().transData
            ))

        ax4 = plt.subplot(1,5,4)
        if self.feat2feat:
            ax4.set_title(f'Aggregated Point Cloud Encoded Feature Map - Real Target Image ',
                           size=20, fontweight='bold')
        elif self.cycle_feat_gan:
            ax4.set_title(f'Reverse Generated Sparse Point Cloud Feature Map - Cycle Source Image ',
                          size=20, fontweight='bold')
        plt.imshow(real_agg_feature_map.numpy(), interpolation=interp, extent=ext,
                   origin=orig, vmax=feature_map_max, vmin=feature_map_min)
        plt.colorbar()
        for box in gt_bboxes_3d.tensor.numpy():
            plt.gca().add_patch(Rectangle(
                box[:2]-box[3:5]/2, box[3], box[4],
                lw=1, ec='tab:orange', fc='tab:orange', alpha=0.25,
                transform=Affine2D().rotate_around(box[0], box[1], -box[6]) + 
                    plt.gca().transData
            ))

        ax5 = plt.subplot(1,5,5)
        ax5.set_title(f'Aggregated Point Cloud', size=20, fontweight='bold')
        plt.scatter(agg_points[0][0][:,0], agg_points[0][0][:,1], s=0.00001, alpha=0.5)
        for box in gt_bboxes_3d.tensor.numpy():
            plt.gca().add_patch(Rectangle(
                box[:2]-box[3:5]/2, box[3], box[4],
                lw=1, ec='tab:orange', fc='tab:orange', alpha=0.25,
                transform=Affine2D().rotate_around(box[0], box[1], -box[6]) + 
                    plt.gca().transData
            ))
        plt.xlim(-54,54)
        plt.ylim(-54,54)

        plt.tight_layout()
        plt.savefig(f'{filename}_gan.png')
        plt.close()

        ##################################################################################

        if self.feat2feat:
            plt.figure(figsize=(15*2,15))
            ax1 = plt.subplot(1,2,1)
            ax1.set_title(f'Fast Fourier Transform of the Mean Fake Aggregated Feature Map',
                          size=20, fontweight='bold')
            plt.imshow(np.log(abs(fft_fake_agg_feature_map)), cmap='viridis')
            plt.colorbar()

            ax2 = plt.subplot(1,2,2)
            ax2.set_title(f'Fast Fourier Transform of the Mean Real Aggregated Feature Map',
                          size=20, fontweight='bold')
            plt.imshow(np.log(abs(fft_real_agg_feature_map)), cmap='viridis')
            plt.colorbar()

            plt.tight_layout()
            plt.savefig(f'{filename}_fft_maps.png')
            plt.close()

            ##################################################################################

            plt.figure(figsize=(15,15))
            ax1 = plt.subplot(1,1,1)
            ax1.set_title(f'Elliptical Gaussian Heatmap Scene Mask', size=20, fontweight='bold')
            plt.imshow(heatmap_masks.numpy(), interpolation=interp, extent=ext, origin=orig)
            plt.colorbar()
            for box in gt_bboxes_3d.tensor.numpy():
                plt.gca().add_patch(Rectangle(
                    box[:2]-box[3:5]/2, box[3], box[4],
                    lw=1, ec='tab:orange', fc='tab:orange', alpha=0.25,
                    transform=Affine2D().rotate_around(box[0], box[1], -box[6]) + 
                        plt.gca().transData
                ))

            plt.tight_layout()
            plt.savefig(f'{filename}_heatmap_scene_mask.png')
            plt.close()

            ##################################################################################

            Z1 = full_real_sparse_feature_map.flatten()
            Y1 = full_fake_agg_feature_map.flatten()
            X1 = full_real_agg_feature_map.flatten()

            data_dict = {'Feat_Map_Value': np.array(list(Z1) + \
                                                    list(Y1) + \
                                                    list(X1)),
                         'Feat_Map': np.array(['Sparse_Encoding']*len(Z1) + \
                                              ['GAN_Translation']*len(Y1) + \
                                              ['Ground_Truth']*len(X1))}
            df = pd.DataFrame(data=data_dict)

            plt.figure(figsize=(15*2,15))
            ax1 = plt.subplot(1,2,1)
            ax1.set_title(f'CDFs for the Encoded, GAN, and Ground Truth Feature Maps',
                          size=20, fontweight='bold')
            sns.ecdfplot(data=df, x='Feat_Map_Value', hue='Feat_Map')
            ks_result1 = scipy.stats.ks_2samp(X1, Y1)
            ax1.text(torch.max(X1)/10, 0.1, str(ks_result1), fontsize=12)


            channels = full_real_agg_feature_map.shape[0]
            heatmap_masks = heatmap_masks.unsqueeze(0).expand(channels,-1,-1)

            Z2 = full_real_sparse_feature_map[heatmap_masks!=1].flatten()
            Y2 = full_fake_agg_feature_map[heatmap_masks!=1].flatten()
            X2 = full_real_agg_feature_map[heatmap_masks!=1].flatten()

            obj_data_dict = {'Feat_Map_Value': np.array(list(Z2) + \
                                                        list(Y2) + \
                                                        list(X2)),
                             'Feat_Map': np.array(['Sparse_Encoding']*len(Z2) + \
                                                  ['GAN_Translation']*len(Y2) + \
                                                  ['Ground_Truth']*len(X2))}
            obj_df = pd.DataFrame(data=obj_data_dict)

            ax2 = plt.subplot(1,2,2)
            ax2.set_title(f'CDFs for the Foreground Objects of the Encoded, GAN, and Ground Truth Feature Maps',
                          size=20, fontweight='bold')
            sns.ecdfplot(data=obj_df, x='Feat_Map_Value', hue='Feat_Map')
            ks_result2 = scipy.stats.ks_2samp(X2, Y2)
            ax1.text(torch.max(X2)/10, 0.1, str(ks_result2), fontsize=12)

            plt.tight_layout()
            plt.savefig(f'{filename}_cdfs.png')
            plt.close()
        elif self.cycle_feat_gan:
            pass
            # plt.figure(figsize=(15*1,15))
            # ax1 = plt.subplot(1,1,1)
            # ax1.set_title(f'Distance Map Between Encoded Feature Map and\nReverse Generated Feature Map',
            #               size=20, fontweight='bold')
            # plt.imshow(gan_dist_map.numpy(), interpolation=interp, extent=ext, origin=orig)
            # plt.colorbar()
            # for box in gt_bboxes_3d.tensor.numpy():
            #     plt.gca().add_patch(Rectangle(
            #         box[:2]-box[3:5]/2, box[3], box[4],
            #         lw=1, ec='tab:orange', fc='tab:orange', alpha=0.25,
            #         transform=Affine2D().rotate_around(box[0], box[1], -box[6]) + 
            #             plt.gca().transData
            #     ))

            # plt.tight_layout()
            # plt.savefig(f'{filename}_dist_maps.png')
            # plt.close()

        ########################################################################
