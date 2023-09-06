# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pandas as pd
import scipy
import seaborn as sns
import torch
from .base_visualizer_hook import BaseVisualizerHook

import mmcv
from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only


@HOOKS.register_module('MMDet3dMMGenPairedVisualizationHook')
class PairedVisualizerHook(BaseVisualizerHook):
    """Paired visualization hook for mmdet3d_mmgen framework.

    In this hook, we use the official api `save_image` in torchvision to save
    the visualization results.

    Args:
        model_name (str): A string indicating the translation model
            being employed. Must be either Feat2FeatGAN or CycleGAN.
    """

    def __init__(self, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert model_name in ['Feat2FeatGAN', 'CycleGan']
        self.paired_translation = True
        self.model_name = model_name

    @master_only
    def paired_translation_visualizations(self, runner):
        """The visualizations to be saved after each train iteration.

        Args:
            runner (object): The runner.
        """
        sparse_points = runner.data_batch['points']
        agg_points = runner.data_batch['agg_points']

        mean_real_sparse_feature_map = runner.outputs[f'results']\
                                                     [f'real_{self.res_name_list[0]}']
        mean_fake_agg_feature_map = runner.outputs[f'results']\
                                                  [f'fake_{self.res_name_list[1]}']
        mean_real_agg_feature_map = runner.outputs[f'results']\
                                                  [f'real_{self.res_name_list[1]}']
        
        full_real_sparse_feature_map = runner.outputs[f'results']\
                                                     [f'full_real_{self.res_name_list[0]}']
        full_fake_agg_feature_map = runner.outputs[f'results']\
                                                  [f'full_fake_{self.res_name_list[1]}']
        full_real_agg_feature_map = runner.outputs[f'results']\
                                                  [f'full_real_{self.res_name_list[1]}']
        
        fft_fake_agg_feature_map = runner.outputs[f'results']\
                                                 [f'fft_fake_{self.res_name_list[1]}']
        fft_real_agg_feature_map = runner.outputs[f'results']\
                                                 [f'fft_real_{self.res_name_list[1]}']
        
        heatmap_masks = runner.outputs[f'results']\
                                      [f'heatmap_masks']
        gt_bboxes_3d = runner.outputs[f'results']\
                                     [f'gt_bboxes_3d']

        filename = self.filename_tmpl.format(runner.iter + 1)
        save_filename = osp.join(self.output_dir, filename)
        mmcv.mkdir_or_exist(self.output_dir)

        vis_data_dir = self.filename_tmpl.format(runner.iter + 1) + '_data'
        self.dump_vis_data(runner.outputs[f'results'], vis_data_dir, filename)

        self.save_visualization(sparse_points.data[0][0], agg_points.data[0][0],
                                mean_real_sparse_feature_map, mean_fake_agg_feature_map,
                                mean_real_agg_feature_map, full_real_sparse_feature_map,
                                full_fake_agg_feature_map, full_real_agg_feature_map,
                                fft_fake_agg_feature_map, fft_real_agg_feature_map,
                                heatmap_masks, gt_bboxes_3d, save_filename)

    @master_only
    def save_visualization(self,
                           sparse_points,
                           agg_points,
                           mean_real_sparse_feature_map,
                           mean_fake_agg_feature_map,
                           mean_real_agg_feature_map,
                           full_real_sparse_feature_map,
                           full_fake_agg_feature_map,
                           full_real_agg_feature_map,
                           fft_fake_agg_feature_map,
                           fft_real_agg_feature_map,
                           heatmap_masks,
                           gt_bboxes_3d,
                           filename):
        ##################################################################################
        # Save visualizations: Plot and save the point clouds and feature maps at all stages.
        ##################################################################################
        # if not self.combine_bool:
        #     agg_points = torch.cat((sparse_points, agg_points), dim=0)

        plt.figure(figsize=(10*2,10))
        ax1 = plt.subplot(1,2,1)
        ax1.set_title(f'Sparse Point Cloud', size=15, fontweight='bold')
        plt.scatter(
            sparse_points[:,0], sparse_points[:,1],
            s=0.001, alpha=0.5
        )
        self.plot_bboxes(gt_bboxes_3d)
        plt.xlim(-54,54)
        plt.ylim(-54,54)

        ax2 = plt.subplot(1,2,2)
        ax2.set_title(f'Aggregated Point Cloud', size=15, fontweight='bold')
        plt.scatter(agg_points[:,0], agg_points[:,1], s=0.00001, alpha=0.5)
        self.plot_bboxes(gt_bboxes_3d)
        plt.xlim(-54,54)
        plt.ylim(-54,54)

        plt.tight_layout()
        plt.savefig(f'{filename}_scatter.png', dpi=800)
        plt.close()

        ##################################################################################
        feature_map_max = torch.max(mean_real_agg_feature_map)
        feature_map_min = torch.min(mean_real_agg_feature_map)

        plt.figure(figsize=(6*5,6))
        ax1 = plt.subplot(1,5,1)
        ax1.set_title(f'Sparse Point Cloud', size=15, fontweight='bold')
        plt.scatter(
            sparse_points[:,0], sparse_points[:,1],
            s=0.001, alpha=0.5
        )
        self.plot_bboxes(gt_bboxes_3d)
        plt.xlim(-54,54)
        plt.ylim(-54,54)

        ax2 = plt.subplot(1,5,2)
        ax2.set_title(f'Sparse Point Cloud Encoded Feature Map \n Real Source Image',
                      size=15, fontweight='bold')
        plt.imshow(mean_real_sparse_feature_map.numpy(), interpolation=self.interp, extent=self.ext,
                   origin=self.orig, vmax=feature_map_max, vmin=feature_map_min)
        plt.colorbar()
        self.plot_bboxes(gt_bboxes_3d)

        ax3 = plt.subplot(1,5,3)
        ax3.set_title(f'{self.model_name} Feature Map \n Fake Target Image', size=15, fontweight='bold')
        plt.imshow(mean_fake_agg_feature_map.numpy(), interpolation=self.interp, extent=self.ext,
                   origin=self.orig, vmax=feature_map_max, vmin=feature_map_min)
        plt.colorbar()
        self.plot_bboxes(gt_bboxes_3d)

        ax4 = plt.subplot(1,5,4)
        ax4.set_title(f'Aggregated Point Cloud Encoded Feature Map \n Real Target Image ',
                        size=15, fontweight='bold')
        plt.imshow(mean_real_agg_feature_map.numpy(), interpolation=self.interp, extent=self.ext,
                   origin=self.orig, vmax=feature_map_max, vmin=feature_map_min)
        plt.colorbar()
        self.plot_bboxes(gt_bboxes_3d)

        ax5 = plt.subplot(1,5,5)
        ax5.set_title(f'Aggregated Point Cloud', size=15, fontweight='bold')
        plt.scatter(agg_points[:,0], agg_points[:,1], s=0.00001, alpha=0.5)
        self.plot_bboxes(gt_bboxes_3d)
        plt.xlim(-54,54)
        plt.ylim(-54,54)

        plt.tight_layout()
        plt.savefig(f'{filename}_gan_results.png', dpi=600)
        plt.close()

        ##################################################################################
        feat_map_err = torch.sum(
            abs(full_real_agg_feature_map - full_fake_agg_feature_map), dim=0)
        feat_map_err = feat_map_err / torch.max(feat_map_err)

        plt.figure(figsize=(8*1,8))
        ax1 = plt.subplot(1,1,1)
        ax1.set_title(f'Absolute Error Map',
                      size=15, fontweight='bold')
        plt.imshow(feat_map_err.numpy(), interpolation=self.interp, extent=self.ext,
                   origin=self.orig)
        plt.colorbar()
        self.plot_bboxes(gt_bboxes_3d)

        plt.tight_layout()
        plt.savefig(f'{filename}_feat_map_error.png', dpi=600)
        plt.close()

        ##################################################################################

        plt.figure(figsize=(8*3,8))
        ax1 = plt.subplot(1,3,1)
        ax1.set_title(f'Sparse Point Cloud Encoded Feature Map \n Real Source Image',
                      size=15, fontweight='bold')
        plt.imshow(mean_real_sparse_feature_map.numpy(), interpolation=self.interp, extent=self.ext,
                   origin=self.orig, vmax=feature_map_max, vmin=feature_map_min)
        plt.colorbar()
        self.plot_bboxes(gt_bboxes_3d)

        ax2 = plt.subplot(1,3,2)
        ax2.set_title(f'{self.model_name} Feature Map \n Fake Target Image', size=15, fontweight='bold')
        plt.imshow(mean_fake_agg_feature_map.numpy(), interpolation=self.interp, extent=self.ext,
                   origin=self.orig, vmax=feature_map_max, vmin=feature_map_min)
        plt.colorbar()
        self.plot_bboxes(gt_bboxes_3d)

        ax3 = plt.subplot(1,3,3)
        ax3.set_title(f'Aggregated Point Cloud Encoded Feature Map \n Real Target Image ',
                        size=15, fontweight='bold')
        plt.imshow(mean_real_agg_feature_map.numpy(), interpolation=self.interp, extent=self.ext,
                   origin=self.orig, vmax=feature_map_max, vmin=feature_map_min)
        plt.colorbar()
        self.plot_bboxes(gt_bboxes_3d)

        plt.tight_layout()
        plt.savefig(f'{filename}_gan_translation.png', dpi=600)
        plt.close()

        ##################################################################################

        plt.figure(figsize=(10*2,10))
        ax1 = plt.subplot(1,2,1)
        ax1.set_title(f'Fast Fourier Transform of the Mean Fake Aggregated Feature Map',
                        size=15, fontweight='bold')
        plt.imshow(np.log(abs(fft_fake_agg_feature_map)), cmap='viridis')
        plt.colorbar()

        ax2 = plt.subplot(1,2,2)
        ax2.set_title(f'Fast Fourier Transform of the Mean Real Aggregated Feature Map',
                        size=15, fontweight='bold')
        plt.imshow(np.log(abs(fft_real_agg_feature_map)), cmap='viridis')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f'{filename}_fft_maps.png', dpi=800)
        plt.close()

        ##################################################################################

        plt.figure(figsize=(10,10))
        ax1 = plt.subplot(1,1,1)
        ax1.set_title(f'Elliptical Gaussian Heatmap Scene Mask', size=15, fontweight='bold')
        plt.imshow(heatmap_masks.numpy(), interpolation=self.interp,
                    extent=self.ext, origin=self.orig)
        plt.colorbar()
        self.plot_bboxes(gt_bboxes_3d)

        plt.tight_layout()
        plt.savefig(f'{filename}_heatmap_scene_mask.png', dpi=800)
        plt.close()

        ##################################################################################

        X1 = full_real_sparse_feature_map[
            (full_fake_agg_feature_map!=0) | (full_real_agg_feature_map!=0)
        ].flatten()
        Y1 = full_fake_agg_feature_map[
            (full_fake_agg_feature_map!=0) | (full_real_agg_feature_map!=0)
        ].flatten()
        Z1 = full_real_agg_feature_map[
            (full_fake_agg_feature_map!=0) | (full_real_agg_feature_map!=0)
        ].flatten()
        X1 = X1[X1<5]
        Y1 = Y1[Y1<5]
        Z1 = Z1[Z1<5]

        data_dict = {'Feat_Map_Value': np.array(list(X1) + \
                                                list(Y1) + \
                                                list(Z1)),
                        'Feat_Map': np.array(['Sparse_Encoding']*len(X1) + \
                                            ['GAN_Translation']*len(Y1) + \
                                            ['Ground_Truth']*len(Z1))}
        df = pd.DataFrame(data=data_dict)

        plt.figure(figsize=(10*2,10))
        ax1 = plt.subplot(1,2,1)
        ax1.set_title(f'CDFs for the Encoded, GAN, and Ground Truth Feature Maps',
                        size=15, fontweight='bold')
        sns.ecdfplot(data=df, x='Feat_Map_Value', hue='Feat_Map')
        ks_result1 = scipy.stats.ks_2samp(Y1, Z1)
        ax1.text(torch.max(Z1)/10, 0.1, str(ks_result1), fontsize=16)


        channels = full_real_agg_feature_map.shape[0]
        heatmap_masks = heatmap_masks.unsqueeze(0).expand(channels,-1,-1)

        X2 = full_real_sparse_feature_map[heatmap_masks!=1].flatten()
        Y2 = full_fake_agg_feature_map[heatmap_masks!=1].flatten()
        Z2 = full_real_agg_feature_map[heatmap_masks!=1].flatten()
        X2 = X2[X2<5]
        Y2 = Y2[Y2<5]
        Z2 = Z2[Z2<5]

        obj_data_dict = {'Feat_Map_Value': np.array(list(X2) + \
                                                    list(Y2) + \
                                                    list(Z2)),
                            'Feat_Map': np.array(['Sparse_Encoding']*len(X2) + \
                                                ['GAN_Translation']*len(Y2) + \
                                                ['Ground_Truth']*len(Z2))}
        obj_df = pd.DataFrame(data=obj_data_dict)

        ax2 = plt.subplot(1,2,2)
        ax2.set_title(f'CDFs for the Foreground Objects of the \n Encoded, GAN, and Ground Truth Feature Maps',
                        size=15, fontweight='bold')
        sns.ecdfplot(data=obj_df, x='Feat_Map_Value', hue='Feat_Map')
        ks_result2 = scipy.stats.ks_2samp(Y2, Z2)
        ax2.text(torch.max(Z2)/10, 0.1, str(ks_result2), fontsize=16)

        plt.tight_layout()
        plt.savefig(f'{filename}_cdfs.png', dpi=800)
        plt.close()

        ########################################################################
