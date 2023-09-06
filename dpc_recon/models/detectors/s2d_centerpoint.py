# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
# import spconv.pytorch as spconv
import torch
import torch.nn as nn

from .util_functions import *
from matplotlib.colors import ListedColormap, BoundaryNorm
from mmcv import ConfigDict
from mmdet.models import DETECTORS
from mmdet3d.core.bbox import box_np_ops, LiDARInstance3DBoxes
from mmdet3d.core.bbox.iou_calculators import bbox_overlaps_nearest_3d
from mmdet3d.models.detectors import CenterPoint
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.ops import spconv as spconv
from dpc_recon.core import get_ellip_gaussian_2D
from dpc_recon.models.losses.util_functions import *
from mmgen.models.builder import build_module
from mmgen.models.common import set_requires_grad


@DETECTORS.register_module()
class S2DCenterPoint(CenterPoint):
    """
    A child class that extends MMDet3d CenterPoint to implement the Sparse2Dense
    model for cross-domain 3D object detection
    """

    def __init__(self,
                 checkpoint_dense=None,
                 loss_sdet=None,
                 *args,
                 **kwargs):
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
        assert checkpoint_dense is not None
        # Build dense model
        kwargs['type'] = 'TeacherSparse2DenseCenterPoint'
        model_cfg_dense = ConfigDict(kwargs)
        self.centerpoint_model_dense = init_model(model_cfg_dense, checkpoint_dense)
        set_requires_grad(self.centerpoint_model_dense, False)
        self.div = len(self.centerpoint_model_dense.pts_bbox_head.task_heads)
        self.centerpoint_model_dense.eval()
        '''
        ################################################################################
        # Build sparse model for processing sparse point clouds
        ################################################################################
        '''
        # Build sparse model
        kwargs['type'] = 'StudentSparse2DenseCenterPoint'
        kwargs['pts_neck']['type'] = 'S2DSECONDFPN'
        # kwargs['type'] = 'Sparse2DenseCenterPoint'
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
                      static_ids=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor]): Points of each sample.
                Defaults to None.
            points_dense (list[torch.Tensor]): Dense scene points of
                each sample with sparse background points. Defaults to None.
            points_fg (list[torch.Tensor]): Dense fg points of
                each sample with sparse background points. Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels
                of 3D boxes. Defaults to None.
            static_ids (list[bools]): Booleans indicating the annotation that
                are quasi-stationary by value 1 and dynamic by value 0.
                Defaults to None.
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
        # img_feats, sparse_pts_feats, F_S_a = self.centerpoint_model_sparse.extract_feat(
            points, img=img, img_metas=img_metas)
        with torch.no_grad():
            self.centerpoint_model_dense.eval()
            _, dense_pts_feats, F_D_a, _ = self.centerpoint_model_dense.extract_feat(
                points_dense, img=img, img_metas=img_metas)
            _, _, F_D_b, voxel_ret = self.centerpoint_model_dense.extract_feat(
                points_fg, img=img, img_metas=img_metas, main_pipeline=False)

        losses = dict()
        if sparse_pts_feats:
            losses_pts = self.forward_pts_train(
                sparse_pts_feats, F_S_a, F_S_b,
                dense_pts_feats, F_D_a, F_D_b,
                gt_bboxes_3d, gt_labels_3d, static_ids,
                img_metas, gt_bboxes_ignore, points,
                points_dense, points_fg, pcr_ret, voxel_ret)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
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
                          static_ids,
                          img_metas,
                          gt_bboxes_ignore,
                          points,
                          points_dense,
                          points_fg,
                          pcr_ret,
                          voxel_ret):
        """Forward function for point cloud branch.

        Args:
            sparse_pts_feats (list[torch.Tensor]): Sparse features of point cloud branch.
            F_S_a (torch.Tensor): Sparse pts_middle_encoder features of point cloud branch.
            F_S_b (torch.Tensor): Sparse S2D features of point cloud branch.
            dense_pts_feats (list[torch.Tensor]): Dense features of point cloud branch.
            F_D_a (torch.Tensor): Dense (points_dense) pts_middle_encoder features of point
                cloud branch.
            F_D_b (torch.Tensor): Dense (points_fg) pts_middle_encoder features of point
                cloud branch.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole.
            static_ids (list[bools]): Booleans indicating the annotation that
                are quasi-stationary by value 1 and dynamic by value 0.
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored.
            points (list[torch.Tensor]): Points of each sample.
            points_dense (list[torch.Tensor]): Dense scene points of
                each sample with sparse background points.
            points_fg (list[torch.Tensor]): Dense fg points of
                each sample with sparse background points.
            pcr_ret (tuple): Data from pcr branch.
            pcr_ret (tuple): Data from voxelization.

        Returns:
            dict: Losses of each branch.
        """
        s2d_losses = dict()

        S_preds = self.centerpoint_model_sparse.pts_bbox_head(sparse_pts_feats)
        with torch.no_grad():
            self.centerpoint_model_dense.eval()
            T_preds = self.centerpoint_model_dense.pts_bbox_head(dense_pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, static_ids, S_preds, T_preds]
        losses, S_preds, T_preds, anno_boxes, inds, masks, cats = \
            self.centerpoint_model_sparse.pts_bbox_head.loss(
                *loss_inputs)
        s2d_losses.update(losses)
        with torch.no_grad():
            s2d_losses['loss_bbox_mean'], s2d_losses['loss_heatmap_mean'] = \
                self.average_centerpoint_losses(losses)

        with torch.no_grad():
            obj_counts = self.static_dynamic_obj_count(static_ids)
            s2d_losses.update(obj_counts)
        
        # self.plotter(points, points_dense, points_fg, gt_bboxes_3d, gt_labels_3d, S_preds, T_preds, static_ids)
        ignore_mask = self.get_ignore_mask(gt_bboxes_3d, static_ids)
        ignore_mask_expanded = ignore_mask.clone().unsqueeze(1).repeat(1,F_D_a.shape[1],1,1)
        # self.ignore_mask_plotter(ignore_mask, gt_bboxes_3d, gt_labels_3d)
        if self.loss_s2d is not None:
            s2d_losses['loss_s2d'] = self.loss_s2d.forward(
                F_S_a, F_D_a, F_S_b=F_S_b, F_D_b=F_D_b, ignore_mask=ignore_mask_expanded.cuda())
                # F_S_a, F_D_a, F_S_b=None, F_D_b=None, ignore_mask=ignore_mask_expanded.cuda())
            # self.s2d_plotter(F_S_a, gt_bboxes_3d, gt_labels_3d, 'F_S_a')
            # self.s2d_plotter(F_D_a, gt_bboxes_3d, gt_labels_3d, 'F_D_a')
            # self.s2d_plotter(F_S_b, gt_bboxes_3d, gt_labels_3d, 'F_S_b')
            # self.s2d_plotter(F_D_b, gt_bboxes_3d, gt_labels_3d, 'F_D_b')
        if self.loss_pcr is not None:
            voxel_features, coors, batch_size, sparse_conv_tensor = voxel_ret
            gen_offset_2, gen_mask_2, gen_offset_4, gen_mask_4 = pcr_ret
            # sparse_shape = self.centerpoint_model_dense.pts_middle_encoder.sparse_shape
            # SparseConvTensor_2 = spconv.SparseConvTensor(
            #     voxel_features, coors, sparse_shape, batch_size)
            # SparseConvTensor_4 = spconv.SparseConvTensor(
            #     voxel_features, coors, sparse_shape, batch_size)
            reconstruction_gt_2 = self.pcr_pool(sparse_conv_tensor[:,:-1,:,:])
            reconstruction_gt_4 = self.pcr_pool(reconstruction_gt_2)
            voxels_tensor = reconstruction_gt_2.reshape(batch_size, 3, 20 * 752 * 752)
            # self.plot_voxel_point_cloud(voxels_tensor,
            #                             gt_bboxes_3d[0].tensor,
            #                             gt_labels_3d[0])
            # self.centerpoint_model_sparse.cp_plotter([points_fg[0]],
            #                                          gt_bboxes_3d[0].tensor,
            #                                          gt_labels_3d[0])
            # reconstruction_gt_4 = self.pcr_pool(self.pcr_pool(SparseConvTensor_4.dense()[:,:-1,:,:]))
            mask_loss, offset_loss = self.loss_pcr.forward(
                reconstruction_gt_2, gen_offset_2, gen_mask_2,
                reconstruction_gt_4, gen_offset_4, gen_mask_4)
            s2d_losses['loss_mask'] = mask_loss
            s2d_losses['loss_offset'] = offset_loss
        if self.loss_distill is not None:
            # losses_iou = self.get_iou_loss(S_preds, img_metas, gt_bboxes_3d)
            # s2d_losses.update(losses_iou)
            losses_distill = self.loss_distill_helper(
                S_preds, T_preds, gt_bboxes_3d, gt_labels_3d,
                static_ids, anno_boxes, inds, masks, cats)
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

    def static_dynamic_obj_count(self, static_ids):
        """Calculate the percentage of objects sampled that are static and
        dynamic for tensorboard"""
        total = 0
        ones = 0
        for idx, ids in enumerate(static_ids):
            ones += torch.sum(ids==1)
            total += len(ids)
        return {'percent_soap': ones/total, 'percent_non_soap': 1-(ones/total)}

    def loss_distill_helper(self, S_preds_dicts, T_preds_dicts, gt_bboxes_3d, gt_labels_3d,
                            static_ids, anno_boxes, inds, masks, cats):
        """Helper function for calculating the distillation loss
        for each head

        Args:
            S_preds_dicts (dict): Output of student forward function
            T_preds_dicts (dict): Output of teacher forward function

        Returns:
            dict: Distillation loss for the heatmaps and teacher-student bboxes
        """
        tasks = []
        tasks_ids = []
        device = gt_labels_3d[0].device
        class_names = self.centerpoint_model_sparse.pts_bbox_head.class_names
        for batch_element in range(len(gt_bboxes_3d)):
            task_masks = []
            flag = 0
            for class_name in class_names:
                task_masks.append([
                    torch.where(gt_labels_3d[batch_element] == class_name.index(i) + flag)
                    for i in class_name
                ])
                flag += len(class_name)

            task_ids = []
            task_boxes = []
            task_classes = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_id = []
                task_box = []
                task_class = []
                for m in mask:
                    task_id.append(static_ids[batch_element][m])
                    task_box.append(gt_bboxes_3d[batch_element][m].tensor)
                    # 0 is background for each task, so we need to add 1 here.
                    task_class.append(gt_labels_3d[batch_element][m] + 1 - flag2)
                task_ids.append(torch.cat(task_id, axis=0).to(device))
                task_boxes.append(torch.cat(task_box, axis=0).to(device))
                task_classes.append(torch.cat(task_class).long().to(device))
                flag2 += len(mask)
            tasks.append(task_boxes)
            tasks_ids.append(task_ids)

        loss_dict = dict()
        loss_hm_distill_mean = None
        # plt.figure(figsize=(16,8))
        # loss_reg_distill_mean = None
        for task_id, (S_preds_dict, T_preds_dict) in enumerate(zip(S_preds_dicts, T_preds_dicts)):

            # S_preds_dict[0]['anno_box'] = torch.cat(
            #     (S_preds_dict[0]['reg'], S_preds_dict[0]['height'],
            #      S_preds_dict[0]['dim'], S_preds_dict[0]['rot']),
            #     dim=1)
            # T_preds_dict[0]['anno_box'] = torch.cat(
            #     (T_preds_dict[0]['reg'], T_preds_dict[0]['height'],
            #      T_preds_dict[0]['dim'], T_preds_dict[0]['rot']),
            #      dim=1)

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

            num_classes = S_preds_dict[0]['heatmap'].shape[1]
            ignore_task_bboxes_3d = [LiDARInstance3DBoxes(t_boxes[task_id]) for t_boxes in tasks]
            ignore_task_static_ids = [id_boxes[task_id] for id_boxes in tasks_ids]
            ignore_task_masks = self.get_ignore_mask(ignore_task_bboxes_3d, ignore_task_static_ids)
            ################################################################################
            # ax = plt.subplot(2,3,task_id+1)
            # ax.set_title(f'Task Specific Ignore Map', size=10, fontweight='bold')
            # plt.imshow(ignore_task_masks[0].detach().cpu().numpy(),
            #     interpolation=self.interp, extent=self.ext, origin=self.orig)
            # plt.colorbar()
            # self.plot_bboxes(ignore_task_bboxes_3d[0])
            ################################################################################
            ignore_task_masks = ignore_task_masks.unsqueeze(1).repeat(1,num_classes,1,1)

            loss_hm_distill = fastfocalloss(
                S_preds_dict[0]['heatmap'], T_preds_dict[0]['heatmap'],
                ind, mask, cat, ignore_task_masks)
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
            with torch.no_grad():
                loss_hm_distill_mean = (loss_hm_distill_mean + loss_hm_distill) \
                    if loss_hm_distill_mean is not None else loss_hm_distill
                # loss_reg_distill_mean = (loss_reg_distill_mean + loss_reg_distill) \
                #     if loss_reg_distill_mean is not None else loss_reg_distill
        # plt.tight_layout()
        # plt.savefig(f'task_specific_ignore_masks.png', dpi=100)
        # plt.close()

        with torch.no_grad():
            loss_dict[f'loss_hm_distill_mean'] = loss_hm_distill_mean / self.div
            # loss_dict[f'loss_reg_distill_mean'] = loss_reg_distill_mean / self.div
        return loss_dict

    def simple_test(self, points, img_metas, img=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):
        """Test function without augmentation."""
        # self.plotter_test(points[0], gt_bboxes_3d[0][0], gt_labels_3d[0][0])
        img_feats, pts_feats, _, _ = self.centerpoint_model_sparse.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.centerpoint_model_sparse.with_pts_bbox:
            bbox_pts = self.centerpoint_model_sparse.simple_test_pts(
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.centerpoint_model_sparse.with_img_bbox:
            bbox_img = self.centerpoint_model_sparse.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def plotter(self, points, points_dense, points_fg, gt_bboxes_3d,
                gt_labels_3d, S_preds, T_preds, static_ids):
        """Function for performing sanity checks."""
        self.interp='None'
        self.ext=[-75.2, 75.2, -75.2, 75.2]
        self.orig='lower'

        plt.figure(figsize=(10*3,10))
        ax1 = plt.subplot(1,3,1)
        ax1.set_title(f'Sparse Point Cloud', size=15, fontweight='bold')
        plt.scatter(
            points[0][:,0].cpu(), points[0][:,1].cpu(),
            s=0.001, alpha=0.5, c=points[0][:,2].cpu(), cmap='viridis',
            vmin=torch.min(points[0][:,2].cpu()), vmax=torch.max(points[0][:,2].cpu())
        )
        self.plot_bboxes(gt_bboxes_3d[0], gt_labels_3d[0])
        plt.xlim(-75.2,75.2)
        plt.ylim(-75.2,75.2)
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])

        ax2 = plt.subplot(1,3,2)
        ax2.set_title(f'SA Static FG + Sparse BG Point Cloud', size=15, fontweight='bold')
        plt.scatter(
            points_dense[0][:,0].cpu(), points_dense[0][:,1].cpu(),
            s=0.001, alpha=0.5, c=points_dense[0][:,2].cpu(), cmap='viridis',
            vmin=torch.min(points[0][:,2].cpu()), vmax=torch.max(points[0][:,2].cpu())
        )
        self.plot_bboxes(gt_bboxes_3d[0][static_ids[0]==1], gt_labels_3d[0][static_ids[0]==1])
        plt.xlim(-75.2,75.2)
        plt.ylim(-75.2,75.2)
        ax2.get_xaxis().set_ticks([])
        ax2.get_yaxis().set_ticks([])

        ax3 = plt.subplot(1,3,3)
        ax3.set_title(f'SA Static FG Point Cloud', size=15, fontweight='bold')
        plt.scatter(
            points_fg[0][:,0].cpu(), points_fg[0][:,1].cpu(),
            s=0.001, alpha=0.5, c=points_fg[0][:,2].cpu(), cmap='viridis',
            vmin=torch.min(points[0][:,2].cpu()), vmax=torch.max(points[0][:,2].cpu())
        )
        self.plot_bboxes(gt_bboxes_3d[0][static_ids[0]==1], gt_labels_3d[0][static_ids[0]==1])
        plt.xlim(-75.2,75.2)
        plt.ylim(-75.2,75.2)
        ax3.get_xaxis().set_ticks([])
        ax3.get_yaxis().set_ticks([])

        plt.tight_layout()
        plt.savefig(f'scatter.png', dpi=100)
        plt.close()

        for i in range(self.div):
            plt.figure(figsize=(10*2,10))
            ax1 = plt.subplot(1,2,1)
            ax1.set_title(f'Student Heatmap-{i}',
                          size=15, fontweight='bold')
            plt.imshow(S_preds[i][0]['heatmap'][0][0].detach().cpu().numpy(),
                interpolation=self.interp, extent=self.ext, origin=self.orig)
            plt.colorbar()
            self.plot_bboxes(gt_bboxes_3d[0], gt_labels_3d[0])

            ax2 = plt.subplot(1,2,2)
            ax2.set_title(f'Teacher Heatmap-{i}',
                          size=15, fontweight='bold')
            plt.imshow(T_preds[i][0]['heatmap'][0][0].cpu().numpy(),
                interpolation=self.interp, extent=self.ext, origin=self.orig)
            plt.colorbar()
            self.plot_bboxes(gt_bboxes_3d[0], gt_labels_3d[0])
            plt.tight_layout()
            plt.savefig(f'hm_results_student_teacher-{i}.png', dpi=100)
            plt.close()
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
        plt.savefig(f'fm_results_{name}.png', dpi=100)
        plt.close()
        return None

    def ignore_mask_plotter(self, mask, gt_bboxes_3d, gt_labels_3d):
        """Function for performing sanity checks for ignore masks."""
        cmap = ListedColormap(['purple', 'yellow'])
        norm = BoundaryNorm([0, 0.5, 1], cmap.N)

        plt.figure(figsize=(10*2,10))
        ax1 = plt.subplot(1,2,1)
        ax1.set_title(f'Ignore Mask - Frame 0',
                      size=15, fontweight='bold')
        plt.imshow(mask[0].detach().cpu().numpy(), cmap=cmap, norm=norm,
            interpolation=self.interp, extent=self.ext, origin=self.orig)
        plt.colorbar()
        self.plot_bboxes(gt_bboxes_3d[0], gt_labels_3d[0])
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])

        ax2 = plt.subplot(1,2,2)
        ax2.set_title(f'Ignore Mask - Frame 1',
                      size=15, fontweight='bold')
        plt.imshow(mask[1].detach().cpu().numpy(), cmap=cmap, norm=norm,
            interpolation=self.interp, extent=self.ext, origin=self.orig)
        plt.colorbar()
        self.plot_bboxes(gt_bboxes_3d[1], gt_labels_3d[1])
        ax2.get_xaxis().set_ticks([])
        ax2.get_yaxis().set_ticks([])

        plt.tight_layout()
        plt.savefig(f'ignore_masks.png', dpi=100)
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
            # transform = Affine2D().rotate_around(*box[:2], -box[6]) + plt.gca().transData
            # rect = plt.Rectangle(box[:2]-box[3:5]/2, width=box[3], height=box[4],
            #                     fc=cmap[label], alpha=float(score)*0.5, transform=transform)
            # plt.gca().add_patch(rect)
            transform = Affine2D().rotate_around(*box[:2], -box[6]) + plt.gca().transData
            rect = plt.Rectangle(box[:2]-box[3:5]/2, width=box[3], height=box[4],
                                ec=cmap[label], alpha=float(score), fill=False,
                                transform=transform, linewidth=2)
            plt.gca().add_patch(rect)
        return None

    def index_plot(self, task_id, S_preds_dict, T_preds_dict, gt_bboxes_3d, gt_labels_3d):
        plt.figure(figsize=(10*2,10))
        ax1 = plt.subplot(1,2,1)
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
        plt.savefig(f'index_map_student_teacher-{task_id}.png', dpi=100)
        plt.close()
        return None

    def get_iou_loss(self, S_preds, img_metas, gt_bboxes_3d):
        bbox_list = self.centerpoint_model_sparse.pts_bbox_head.get_bboxes(
            S_preds, img_metas, rescale=False)
        pred_bboxes_3d = [boxes[0] for boxes in bbox_list]
        ret = dict()
        ious_list = []
        for idx, (pred_bbox_3d, gt_bbox_3d) in enumerate(zip(pred_bboxes_3d, gt_bboxes_3d)):
            bbox_ious = bbox_overlaps_nearest_3d(pred_bbox_3d.tensor.cuda(),
                                                 gt_bbox_3d.tensor.cuda(),
                                                 mode='giou')
            ious = torch.max(bbox_ious, 1).values
            ious_list.append(ious)
        ret[f'losses_iou'] = list(map(
            lambda ious: torch.mean((1 - ious)).view(1,),
            ious_list
            ))[0]
        return ret

    def plotter_test(self, points, gt_bboxes_3d, gt_labels_3d):
        """Function for performing sanity checks."""
        import datetime
        import matplotlib.pyplot as plt
        now = datetime.datetime.now()
        time_string = now.strftime("%H:%M:%S")
        self.interp='None'
        self.ext=[-75.2, 75.2, -75.2, 75.2]
        self.orig='lower'

        plt.figure(figsize=(10,10))
        ax1 = plt.subplot(1,1,1)
        ax1.set_title(f'PCL2 Point Cloud', size=15, fontweight='bold')
        plt.scatter(
            points[:,0].cpu(), points[:,1].cpu(),
            s=0.001, alpha=0.5, c=points[:,2].cpu(), cmap='viridis',
            vmin=torch.min(points[:,2].cpu()), vmax=torch.max(points[:,2].cpu())
        )
        self.plot_bboxes(gt_bboxes_3d.tensor, gt_labels_3d)
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])
        plt.xlim(-75.2,75.2)
        plt.ylim(-75.2,75.2)

        plt.tight_layout()
        plt.savefig(f'scatter_test_{time_string}.png', dpi=100)
        plt.close()

        return None

    def get_ignore_mask(self, gt_bboxes_3d, static_ids):
        """
        Function for generating a binary tensor with zeros in regions
        containing dynamic objects and 1 elsewhere
        """
        bboxes_3d = [
            torch.cat(
                (boxes_3d.gravity_center, boxes_3d.tensor[:, 3:]), dim=1
                ).to(boxes_3d.device) for boxes_3d in gt_bboxes_3d
        ]

        ignore_mask = bboxes_3d[0].new_zeros(
            (len(bboxes_3d), self.feature_map_size[1], self.feature_map_size[0]))
        
        for idx, frame_boxes in enumerate(bboxes_3d):
            assert len(frame_boxes)==len(static_ids[idx])
            for obj_id, box_3d in enumerate(frame_boxes):
                if static_ids[idx][obj_id]==1:
                    continue
                device = box_3d.device
                x, y = box_3d[0], box_3d[1]

                coor_x = (
                    x - self.pc_range[0]
                ) / self.voxel_size[0] / self.out_size_factor
                coor_y = (
                    y - self.pc_range[1]
                ) / self.voxel_size[1] / self.out_size_factor

                center = torch.tensor([coor_x, coor_y],
                                       dtype=torch.float32,
                                       device=device)
                center_int = center.to(torch.int32)
                
                width_int = torch.tensor([box_3d[3] * 1.2], dtype=torch.float32, device=device)
                length_int = torch.tensor([box_3d[4] * 1.2], dtype=torch.float32, device=device)
                yaw_deg = float(box_3d[6].to(torch.float32)) * (180 / np.pi)

                get_ellip_gaussian_2D(ignore_mask[idx],
                                      center_int,
                                      width_int,
                                      length_int,
                                      yaw_deg,
                                      k=1,
                                      gt_bboxes_3d=frame_boxes,
                                      obj_num=obj_id)
        return 1 - ignore_mask

    def plot_voxel_point_cloud(self, tensor, gt_bboxes_3d, gt_labels_3d):
        """Function for performing sanity checks."""
        tensor = tensor.cpu()[0]
        tensor = tensor[:, tensor.sum(dim=0)!=0]
        interp='None'
        ext=[-75.2/2, 75.2/2, -75.2/2, 75.2/2]
        orig='lower'

        plt.figure(figsize=(10,10))
        ax1 = plt.subplot(1,1,1)
        ax1.set_title(f'2D Voxel Point Cloud', size=15, fontweight='bold')
        plt.scatter(
            tensor[0,:], tensor[1,:],
            s=0.01, alpha=0.5, c=tensor[2, :], cmap='viridis',
        )
        # self.plot_bboxes(gt_bboxes_3d, gt_labels_3d)
        plt.xlim(-75.2/2,75.2/2)
        plt.ylim(-75.2/2,75.2/2)

        plt.tight_layout()
        plt.savefig(f'2d_voxel_point_cloud.png', dpi=100)
        plt.close()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'3D Voxel Point Cloud', size=15, fontweight='bold')
        ax.scatter(tensor[0, :], tensor[1, :], tensor[2, :],
                c=tensor[2, :], marker='o', s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.savefig(f'3d_voxel_point_cloud.png', dpi=100)
        plt.close()

        return None
