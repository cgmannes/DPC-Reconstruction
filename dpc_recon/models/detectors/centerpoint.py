# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import os
import torch

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from mmdet3d.models import CenterPoint as _CenterPoint


@DETECTORS.register_module(force=True)
class CenterPoint(_CenterPoint):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CenterPoint,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained, init_cfg)

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x, _ = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
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
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton.

        The function implementation process is as follows:

            - step 1: map features back for double-flip augmentation.
            - step 2: merge all features and generate boxes.
            - step 3: map boxes back for scale augmentation.
            - step 4: merge results.

        Args:
            feats (list[torch.Tensor]): Feature of point cloud.
            img_metas (list[dict]): Meta information of samples.
            rescale (bool): Whether to rescale bboxes. Default: False.

        Returns:
            dict: Returned bboxes consists of the following keys:

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): Predicted bboxes.
                - scores_3d (torch.Tensor): Scores of predicted boxes.
                - labels_3d (torch.Tensor): Labels of predicted boxes.
        """
        # only support aug_test for one sample
        outs_list = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            # merge augmented outputs before decoding bboxes
            for task_id, out in enumerate(outs):
                for key in out[0].keys():
                    if img_meta[0]['pcd_horizontal_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[2])
                        if key == 'reg':
                            outs[task_id][0][key][:, 1, ...] = 1 - outs[
                                task_id][0][key][:, 1, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                    if img_meta[0]['pcd_vertical_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[3])
                        if key == 'reg':
                            outs[task_id][0][key][:, 0, ...] = 1 - outs[
                                task_id][0][key][:, 0, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]

            outs_list.append(outs)

        preds_dicts = dict()
        scale_img_metas = []

        # concat outputs sharing the same pcd_scale_factor
        for i, (img_meta, outs) in enumerate(zip(img_metas, outs_list)):
            pcd_scale_factor = img_meta[0]['pcd_scale_factor']
            if pcd_scale_factor not in preds_dicts.keys():
                preds_dicts[pcd_scale_factor] = outs
                scale_img_metas.append(img_meta)
            else:
                for task_id, out in enumerate(outs):
                    for key in out[0].keys():
                        preds_dicts[pcd_scale_factor][task_id][0][key] += out[
                            0][key]

        aug_bboxes = []

        for pcd_scale_factor, preds_dict in preds_dicts.items():
            for task_id, pred_dict in enumerate(preds_dict):
                # merge outputs with different flips before decoding bboxes
                for key in pred_dict[0].keys():
                    preds_dict[task_id][0][key] /= len(outs_list) / len(
                        preds_dicts.keys())
            bbox_list = self.pts_bbox_head.get_bboxes(
                preds_dict, img_metas[0], rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        if len(preds_dicts.keys()) > 1:
            # merge outputs with different scales after decoding bboxes
            merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, scale_img_metas,
                                                self.pts_bbox_head.test_cfg)
            return merged_bboxes
        else:
            for key in bbox_list[0].keys():
                bbox_list[0][key] = bbox_list[0][key].to('cpu')
            return bbox_list[0]

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            pts_bbox = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=pts_bbox)
        return [bbox_list]

    # def forward(self, points, gt_bboxes_3d, gt_labels_3d, **kwargs):
    #     self.cp_plotter(points, gt_bboxes_3d, gt_labels_3d)
    #     print('CenterPoint Exit')
    #     exit()
    #     return None

    def cp_plotter(self, points, gt_bboxes_3d, gt_labels_3d):
        """Function for performing sanity checks."""
        import datetime
        factor = 1
        now = datetime.datetime.now()
        time_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        self.interp='None'
        self.ext=[-75.2/factor, 75.2/factor, -75.2/factor, 75.2/factor]
        self.orig='lower'

        # DENSITY = f'sparse'.capitalize()
        DENSITY = f'dense'.capitalize()
        # DATASET = f'nu' + f'scenes'.capitalize()
        DATASET = f'waymo'.capitalize()
        plt.figure(figsize=(10,10))
        ax1 = plt.subplot(1,1,1)
        ax1.set_title(
            f'{DENSITY} {DATASET} Point Cloud',
            size=15, fontweight='bold',
        )
        plt.scatter(
            points[0][:,0].cpu(), points[0][:,1].cpu(),
            s=0.001, alpha=0.5, c=points[0][:,2].cpu(), cmap='viridis',
            vmin=torch.min(points[0][:,2].cpu()), vmax=torch.max(points[0][:,2].cpu())
        )
        # self.plot_bboxes(gt_bboxes_3d, gt_labels_3d)
        self.plot_bboxes(gt_bboxes_3d[0].tensor, gt_labels_3d[0])
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])
        plt.xlim(-75.2/factor,75.2/factor)
        plt.ylim(-75.2/factor,75.2/factor)

        DIR = f'{DENSITY.lower()}_{DATASET.lower()}_plots'
        plt.tight_layout()
        os.makedirs(f'{DIR}', exist_ok=True)
        plt.savefig(f'{DIR}/bev_scatter_point_cloud_{time_string}.png', dpi=100)
        plt.close()

        # fig = plt.figure(figsize=(10,10))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_title(f'3d Scene Point Cloud', size=12, fontweight='bold')
        # ax.scatter(points[0][:,0].cpu(), points[0][:,1].cpu(), points[0][:,2].cpu(),
        #            c=points[0][:,2].cpu(), marker='o', s=1)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_axis('scaled')
        # plt.tight_layout()
        # plt.savefig(f'scatter_3d_train_t1_pc2_pipeline_{time_string}.png', dpi=100)
        # plt.close()

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
                                ec=cmap[label], alpha=float(score), fill=False,
                                transform=transform, linewidth=1)
            plt.gca().add_patch(rect)
        return None