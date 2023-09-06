# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn

from .util_functions import *
from mmcv import ConfigDict
from mmdet.models import DETECTORS
from mmdet3d.models.detectors import CenterPoint
from dpc_recon.models.losses.util_functions import *
from mmgen.models.builder import build_module
from mmgen.models.common import set_requires_grad


@DETECTORS.register_module()
class S2DCenterPointBaseline(CenterPoint):
    """
    A child class that extends MMDet3d CenterPoint to implement the Sparse2Dense
    model for in-domain 3D object detection
    """

    def __init__(self,
                 checkpoint_dense=None,
                 loss_sdet=None,
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        '''
        ################################################################################
        # Build Sparse2Dense loss modules
        ################################################################################
        '''
        if loss_sdet is not None:
            self.build_sdet_loss_modules(loss_sdet)
        '''
        ################################################################################
        # Build trained dense model for processing dense point clouds
        ################################################################################
        '''
        # Build dense model
        s2d_training = kwargs['pts_neck'].pop('training')
        s2d_pcr = kwargs['pts_neck'].pop('pcr')
        assert checkpoint_dense is not None
        kwargs['type'] = 'TeacherSparse2DenseCenterPoint'
        kwargs['pts_neck']['type'] = 'SECONDFPN'
        model_cfg_dense = ConfigDict(kwargs)
        self.centerpoint_model_dense = init_model(model_cfg_dense, checkpoint_dense)
        set_requires_grad(self.centerpoint_model_dense, False)
        self.div = len(self.centerpoint_model_dense.pts_bbox_head.task_heads)
        self.centerpoint_model_dense.eval()

        # Build sparse model
        kwargs['type'] = 'StudentSparse2DenseCenterPoint'
        kwargs['pts_voxel_layer']['max_num_points'] = int(kwargs['pts_voxel_layer']['max_num_points'] / 2)
        kwargs['pts_voxel_layer']['max_voxels'] = (int(kwargs['pts_voxel_layer']['max_voxels'][0] / 2),
                                                   int(kwargs['pts_voxel_layer']['max_voxels'][1] / 2))
        kwargs['pts_voxel_layer']['deterministic'] = True
        kwargs['pts_neck']['type'] = 'S2DSECONDFPN'
        kwargs['pts_neck']['training'] = s2d_training
        kwargs['pts_neck']['pcr'] = s2d_pcr
        model_cfg_sparse = ConfigDict(kwargs)
        self.centerpoint_model_sparse = init_model(model_cfg_sparse, checkpoint_dense)

        # Parameters for the ignore mask
        self.grid_size = torch.tensor([1504, 1504])
        self.pc_range = torch.tensor([-75.2, -75.2, -2, 75.2, 75.2, 4])
        self.voxel_size = torch.tensor([0.1, 0.1, 0.15])
        self.out_size_factor = 8
        self.feature_map_size = self.grid_size // self.out_size_factor
        self.interp='None'
        self.ext=[-75.2, 75.2, -75.2, 75.2]
        self.orig='lower'
        self.pcr_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def build_sdet_loss_modules(self, loss_sdet):
        """Build sparse detector loss modules"""

        # Build the loss modules of Sparse2Dense
        self.loss_s2d = build_module(loss_sdet['loss_s2d']) \
            if loss_sdet['loss_s2d'] is not None else None
        self.loss_pcr = build_module(loss_sdet['loss_pcr']) \
            if loss_sdet['loss_pcr'] is not None else None
        self.loss_distill = build_module(loss_sdet['loss_distill']) \
            if loss_sdet['loss_distill'] is not None else None

        return None

    def forward_train(self,
                      points=None,
                      points_dense=None,
                      points_fg=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            points_dense (list[torch.Tensor], optional): Dense fg points of
                each sample with sparse background points. Defaults to None.
            points_fg (list[torch.Tensor], optional): Dense fg points of
                each sample with no background points. Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, sparse_pts_feats, F_S_a, F_S_b, pcr_ret = self.centerpoint_model_sparse.extract_feat(
            points, img=img, img_metas=img_metas,
            )
        with torch.no_grad():
            self.centerpoint_model_dense.eval()
            _, dense_pts_feats, F_D_a, _ = self.centerpoint_model_dense.extract_feat(
                points_dense, img=img, img_metas=img_metas,
                )
            _, _, F_D_b, voxel_ret = self.centerpoint_model_dense.extract_feat(
                points_fg, img=img, img_metas=img_metas, main_pipeline=False,
                )

        losses = dict()
        losses_pts = self.forward_pts_train(
            sparse_pts_feats, F_S_a, F_S_b,
            dense_pts_feats, F_D_a, F_D_b,
            gt_bboxes_3d, gt_labels_3d, img_metas,
            gt_bboxes_ignore, points, points_dense,
            points_fg, pcr_ret, voxel_ret,
            )
        losses.update(losses_pts)
        return losses

    def forward_pts_train(self,
                          sparse_pts_feats,
                          F_S_a,
                          F_S_b,
                          dense_pts_feats,
                          F_D_a,
                          F_D_b,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore,
                          points,
                          points_dense,
                          points_fg,
                          pcr_ret,
                          voxel_ret,
                          ):
        """Forward function for point cloud branch.

        Args:
            sparse_pts_feats (list[torch.Tensor]): Sparse features of point cloud branch
            F_S_a (torch.Tensor): Sparse pts_middle_encoder features of the FG only branch
            F_S_b (torch.Tensor): Sparse pts_middle_encoder features of the FG+BG branch
            dense_pts_feats (list[torch.Tensor]): Dense features of point cloud branch
            F_D_a (torch.Tensor): Dense pts_middle_encoder features of the FG only branch
            F_D_b (torch.Tensor): Dense pts_middle_encoder features of the FG+BG branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        s2d_losses = dict()

        S_preds = self.centerpoint_model_sparse.pts_bbox_head(sparse_pts_feats)

        with torch.no_grad():
            self.centerpoint_model_dense.eval()
            T_preds = self.centerpoint_model_dense.pts_bbox_head(dense_pts_feats)
        
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, S_preds, T_preds]
        losses, S_preds, T_preds, anno_boxes, inds, masks, cats = \
            self.centerpoint_model_sparse.pts_bbox_head.loss(
                *loss_inputs
                )
        
        s2d_losses.update(losses)
        # with torch.no_grad():
        #     s2d_losses['loss_bbox_mean'], s2d_losses['loss_heatmap_mean'] = \
        #         self.average_centerpoint_losses(losses)

        # self.plotter(points, points_dense, points_fg, gt_bboxes_3d, gt_labels_3d, S_preds, T_preds)
        if self.loss_s2d is not None:
            s2d_losses['loss_s2d'] = self.loss_s2d.forward(
                F_S_a, F_D_a, F_S_b=F_S_b, F_D_b=F_D_b,
                )
        if self.loss_pcr is not None:
            voxel_features, coors, batch_size, sparse_conv_tensor = voxel_ret
            gen_offset_2, gen_mask_2, gen_offset_4, gen_mask_4 = pcr_ret

            reconstruction_gt_2 = self.pcr_pool(sparse_conv_tensor[:,:-1,:,:])
            reconstruction_gt_4 = self.pcr_pool(reconstruction_gt_2)
            # voxels_tensor = reconstruction_gt_2.reshape(batch_size, 3, 20 * 752 * 752)
            # self.plot_voxel_point_cloud(voxels_tensor,
            #                             gt_bboxes_3d[0].tensor,
            #                             gt_labels_3d[0])
            # self.centerpoint_model_sparse.cp_plotter([points_fg[0]],
            #                                          gt_bboxes_3d[0].tensor,
            #                                          gt_labels_3d[0])
            # reconstruction_gt_4 = self.pcr_pool(self.pcr_pool(SparseConvTensor_4.dense()[:,:-1,:,:]))
            mask_loss, offset_loss = self.loss_pcr.forward(
                reconstruction_gt_2, gen_offset_2, gen_mask_2,
                reconstruction_gt_4, gen_offset_4, gen_mask_4,
                )
            s2d_losses['loss_mask'] = mask_loss
            s2d_losses['loss_offset'] = offset_loss
        if self.loss_distill is not None:
            losses_distill = self.loss_distill_helper(
                S_preds, T_preds, anno_boxes, inds, masks, cats,
                )
            s2d_losses.update(losses_distill)

        return s2d_losses

    def average_centerpoint_losses(self, losses):
        """Average the bbox losses and the heatmap losses of CenterPoint
        over all the heads"""
        loss_bbox_mean = None
        loss_heatmap_mean = None
        for k, v in losses.items():
            if k.find('bbox')!=-1:
                loss_bbox_mean = (loss_bbox_mean + v) \
                    if loss_bbox_mean is not None else v
            if k.find('heatmap')!=-1:
                loss_heatmap_mean = (loss_heatmap_mean + v) \
                    if loss_heatmap_mean is not None else v
        return (loss_bbox_mean / self.div), (loss_heatmap_mean / self.div)

    def loss_distill_helper(self, S_preds_dicts, T_preds_dicts,
                            anno_boxes, inds, masks, cats,
                            ):
        """Helper function for calculating the distillation loss
        for each head

        Args:
            S_preds_dicts (dict): Output of student forward function
            T_preds_dicts (dict): Output of teacher forward function

        Returns:
            dict: Distillation loss for the heatmaps and teacher-student bboxes
        """
        loss_dict = dict()
        loss_hm_distill_mean = None
        # loss_reg_distill_mean = None
        for task_id, (S_preds_dict, T_preds_dict) in enumerate(zip(S_preds_dicts, T_preds_dicts)):

            S_preds_dict[0]['anno_box'] = torch.cat(
                (S_preds_dict[0]['reg'], S_preds_dict[0]['height'],
                 S_preds_dict[0]['dim'], S_preds_dict[0]['rot']),
                dim=1)
            T_preds_dict[0]['anno_box'] = torch.cat(
                (T_preds_dict[0]['reg'], T_preds_dict[0]['height'],
                 T_preds_dict[0]['dim'], T_preds_dict[0]['rot']),
                 dim=1)

            ind = inds[task_id]
            target_box = anno_boxes[task_id]
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan
            cat = cats[task_id]

            # for i in ind[0]:
            #     row = i // 188
            #     col = i % 188
            #     S_preds_dict[0]['heatmap'][0][0][row, col] = 100
            #     T_preds_dict[0]['heatmap'][0][0][row, col] = 100
            # self.index_plot(task_id, S_preds_dict, T_preds_dict, gt_bboxes_3d, gt_labels_3d)
            # exit()

            loss_hm_distill = fastfocalloss(
                S_preds_dict[0]['heatmap'], T_preds_dict[0]['heatmap'],
                ind, mask, cat)
            # loss_reg_distill = distill_reg_loss(
            #     S_preds_dict[0]['anno_box'], T_preds_dict[0]['anno_box'],
            #     mask, ind)
            # loss_hm_distill, loss_reg_distill = self.loss_distill.forward(
            #     S_preds_dict, T_preds_dict, ind, mask, cat)

            # code_weights = self.centerpoint_model_sparse.pts_bbox_head.train_cfg.get('code_weights', None)
            # loss_reg_distill = (loss_reg_distill * loss_reg_distill.new_tensor(
            #     code_weights)).sum() * self.centerpoint_model_sparse.pts_bbox_head.weight

            loss_dict[f'task{task_id}.loss_hm_distill'] = loss_hm_distill
            # loss_dict[f'task{task_id}.loss_reg_distill'] = loss_reg_distill
            # with torch.no_grad():
            #     loss_hm_distill_mean = (loss_hm_distill_mean + loss_hm_distill) \
            #         if loss_hm_distill_mean is not None else loss_hm_distill
                # loss_reg_distill_mean = (loss_reg_distill_mean + loss_reg_distill) \
                #     if loss_reg_distill_mean is not None else loss_reg_distill

        # with torch.no_grad():
        #     loss_dict[f'loss_hm_distill_mean'] = loss_hm_distill_mean / self.div
            # loss_dict[f'loss_reg_distill_mean'] = loss_reg_distill_mean / self.div
        return loss_dict

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentation."""
        img_feats, pts_feats, _, _, _ = self.centerpoint_model_sparse.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.centerpoint_model_sparse.simple_test_pts(
            pts_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def plotter(self, points, points_dense, points_fg, gt_bboxes_3d, gt_labels_3d, S_preds, T_preds):
        """Function for performing sanity checks."""
        import datetime
        now = datetime.datetime.now()
        time_string = now.strftime("%Y-%m-%d-%H-%M-%S")

        plt.figure(figsize=(10*3,10))
        ax1 = plt.subplot(1,3,1)
        ax1.set_title(f'Sparse Point Cloud', size=15, fontweight='bold')
        plt.scatter(
            points[0][:,0].cpu(), points[0][:,1].cpu(), s=0.001,
            alpha=0.5, c=points[0][:,2].cpu(), cmap='viridis',
            vmin=torch.min(points[0][:,2].cpu()),
            vmax=torch.max(points[0][:,2].cpu()),
        )
        self.plot_bboxes(gt_bboxes_3d[0], gt_labels_3d[0])
        plt.xlim(-72.5,72.5)
        plt.ylim(-72.5,72.5)

        ax2 = plt.subplot(1,3,2)
        ax2.set_title(f'Aggregated Point Cloud', size=15, fontweight='bold')
        plt.scatter(
            points_dense[0][:,0].cpu(), points_dense[0][:,1].cpu(), s=0.001,
            alpha=0.5, c=points_dense[0][:,2].cpu(), cmap='viridis',
            vmin=torch.min(points_dense[0][:,2].cpu()),
            vmax=torch.max(points_dense[0][:,2].cpu()),
        )
        self.plot_bboxes(gt_bboxes_3d[0], gt_labels_3d[0])
        plt.xlim(-72.5,72.5)
        plt.ylim(-72.5,72.5)

        ax3 = plt.subplot(1,3,3)
        ax3.set_title(f'Aggregated Foreground Point Cloud', size=15, fontweight='bold')
        plt.scatter(
            points_fg[0][:,0].cpu(), points_fg[0][:,1].cpu(), s=0.001,
            alpha=0.5, c=points_fg[0][:,2].cpu(), cmap='viridis',
            vmin=torch.min(points_fg[0][:,2].cpu()),
            vmax=torch.max(points_fg[0][:,2].cpu()),
        )
        self.plot_bboxes(gt_bboxes_3d[0], gt_labels_3d[0])
        plt.xlim(-72.5,72.5)
        plt.ylim(-72.5,72.5)

        # DATASET = f'nu' + f'scenes'
        DATASET = f'waymo'
        DIR = f's2d_{DATASET}_plots'
        plt.tight_layout()
        os.makedirs(f'{DIR}', exist_ok=True)
        plt.savefig(f'{DIR}/bev_scatter_point_cloud_{time_string}.png', dpi=300)
        plt.close()

        # for i in range(self.div):
        #     plt.figure(figsize=(10*2,10))
        #     ax1 = plt.subplot(1,2,1)
        #     ax1.set_title(f'Student Heatmap-{i}',
        #                   size=15, fontweight='bold')
        #     plt.imshow(S_preds[i][0]['heatmap'][0][0].detach().cpu().numpy(),
        #         interpolation=self.interp, extent=self.ext, origin=self.orig)
        #     plt.colorbar()
        #     self.plot_bboxes(gt_bboxes_3d[0], gt_labels_3d[0])

        #     ax2 = plt.subplot(1,2,2)
        #     ax2.set_title(f'Teacher Heatmap-{i}',
        #                   size=15, fontweight='bold')
        #     plt.imshow(T_preds[i][0]['heatmap'][0][0].cpu().numpy(),
        #         interpolation=self.interp, extent=self.ext, origin=self.orig)
        #     plt.colorbar()
        #     self.plot_bboxes(gt_bboxes_3d[0], gt_labels_3d[0])
        #     plt.tight_layout()
        #     plt.savefig(f'./s2d_student_plots/hm_results_student_teacher-{i}.png', dpi=300)
        #     plt.close()
        return None

    def s2d_plotter(self, feats, gt_bboxes_3d, gt_labels_3d, name):
        """Function for performing sanity checks."""

        inds = feats > 0
        plt.figure(figsize=(10*2,10))
        ax1 = plt.subplot(1,2,1)
        ax1.set_title(f'S2D Index Map - {name}',
                      size=15, fontweight='bold')
        inds = inds.any(dim=1, keepdim=True)
        plt.imshow(inds[0][0].detach().cpu().numpy(),
            interpolation=self.interp, extent=self.ext, origin=self.orig)
        plt.colorbar()
        self.plot_bboxes(gt_bboxes_3d[0], gt_labels_3d[0])

        ax2 = plt.subplot(1,2,2)
        ax2.set_title(f'S2D Feature Map - {name}',
                      size=15, fontweight='bold')
        feats = torch.max(feats, dim=1, keepdim=True)
        plt.imshow(feats.values[0][0].detach().cpu().numpy(),
            interpolation=self.interp, extent=self.ext, origin=self.orig)
        plt.colorbar()
        self.plot_bboxes(gt_bboxes_3d[0], gt_labels_3d[0])
        plt.tight_layout()
        plt.savefig(f'./s2d_student_plots/fm_results_{name}.png', dpi=300)
        plt.close()
        return None

    def plot_bboxes(self, boxes, labels=None, scores=None, cmap=['tab:blue', 'tab:purple', 'tab:green', 'tab:orange', 'tab:pink', 'tab:red']):
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Affine2D

        if labels is None:
            labels = [0] * len(boxes)
        if scores is None:
            scores = [1] * len(boxes)
        for box, label, score in zip(boxes, labels, scores):
            transform = Affine2D().rotate_around(*box[:2], -box[6]) + plt.gca().transData
            rect = plt.Rectangle(box[:2]-box[3:5]/2, width=box[3], height=box[4],
                                fc=cmap[label], alpha=float(score)*0.5, transform=transform)
            plt.gca().add_patch(rect)
            rect = plt.Rectangle(box[:2]-box[3:5]/2, width=box[3], height=box[4],
                                ec=cmap[label], alpha=float(score), fill=False, transform=transform)
            plt.gca().add_patch(rect)
        return None

    def index_plot(self, task_id, S_preds_dict, T_preds_dict, gt_bboxes_3d, gt_labels_3d):
        plt.figure(figsize=(10*2,10))
        ax1 = plt.subplot(1,1,1)
        ax1.set_title(f'Student Index Heatmap-{task_id}',
                        size=15, fontweight='bold')
        plt.imshow(S_preds_dict[0]['heatmap'][0][0].detach().cpu().numpy(),
            interpolation=self.interp, extent=self.ext, origin=self.orig)
        plt.colorbar()
        self.plot_bboxes(gt_bboxes_3d[0], gt_labels_3d[0])

        ax2 = plt.subplot(1,2,2)
        ax2.set_title(f'Teacher Index Heatmap-{task_id}',
                        size=15, fontweight='bold')
        plt.imshow(T_preds_dict[0]['heatmap'][0][0].cpu().numpy(),
            interpolation=self.interp, extent=self.ext, origin=self.orig)
        plt.colorbar()
        self.plot_bboxes(gt_bboxes_3d[0], gt_labels_3d[0])

        plt.tight_layout()
        os.makedirs('./s2d_student_plots', exist_ok=True)
        plt.savefig(f'./s2d_student_plots/index_map_student_teacher-{task_id}.png', dpi=300)
        plt.close()
        return None