# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
# import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import DETECTORS
from mmdet3d.models.detectors import CenterPoint
from mmdet3d.ops import spconv as spconv


@DETECTORS.register_module()
class GOATCenterPoint(CenterPoint):
    """
    A child class that extends MMDet3d CenterPoint to implement a general
    voxelized point cloud reconstruction model consisting the the CenterPoint
    encoder module and the Sparse2Dense point cloud reconstruction module
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Parameters for the ignore mask
        self.grid_size = torch.tensor([1504, 1504])
        self.pc_range = torch.tensor([-75.2, -75.2, -2, 75.2, 75.2, 4])
        self.voxel_size = torch.tensor([0.1, 0.1, 0.15])
        self.out_size_factor = 8
        self.feature_map_size = self.grid_size // self.out_size_factor
        self.interp='None'
        self.ext=[-75.2, 75.2, -75.2, 75.2]
        self.orig='lower'

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points"""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x, _ = self.pts_middle_encoder(voxel_features, coors, batch_size)
        N, _, H, W = x.shape
        F = self.pts_neck.forward_s2d(x)
        if self.with_pts_neck:
            pcr_ret = self.pts_neck.forward_pcr(F, training=True, N=N, _=_, H=H, W=W)
        return pcr_ret

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points"""
        img_feats = self.extract_img_feat(img, img_metas)
        pcr_ret = self.extract_pts_feat(points, img_feats, img_metas)
        return img_feats, pcr_ret

    def fg_encoder(self, pts):
        """Extract features of points"""
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        _, input_sp_tensor = self.pts_middle_encoder(voxel_features, coors, batch_size)
        return input_sp_tensor.dense()

    def forward_train(self,
                      points=None,
                      points_fg=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor]): Points of each sample.
                Defaults to None.
            points_fg (list[torch.Tensor]): Dense fg points of
                each sample with sparse background points. Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels
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
        # Dense full point cloud
        img_feats, pcr_ret = self.extract_feat(
            points, img=img, img_metas=img_metas)
        
        # Dense FG point cloud
        gt_reconstruction = self.fg_encoder(points_fg)

        losses = dict()
        if pcr_ret:
            losses_pts = self.forward_pts_train(
                pcr_ret, gt_reconstruction,
                gt_bboxes_3d, gt_labels_3d,
                img_metas, gt_bboxes_ignore,
                points, points_fg)
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
                          pcr_ret,
                          gt_reconstruction,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore,
                          points,
                          points_fg):
        """Forward function for point cloud branch.

        Args:
            pcr_ret (list[torch.Tensor]): Sparse features of point cloud branch.
            gt_reconstruction (torch.Tensor): Sparse pts_middle_encoder features of point cloud branch.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole.
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored.
            points (list[torch.Tensor]): Points of each sample.
            points_fg (list[torch.Tensor]): Dense fg points of
                each sample with sparse background points.

        Returns:
            dict: Losses of each branch.
        """
        pcr_losses = dict()

        # Sanity check
        # self.plotter(points, points_fg, gt_bboxes_3d, gt_labels_3d)

        #PCR loss
        gen_offset_2, gen_mask_2, gen_offset_4, gen_mask_4 = pcr_ret
        reconstruction_gt_2 = self.pcr_pool(gt_reconstruction[:,:-1,:,:])
        reconstruction_gt_4 = self.pcr_pool(reconstruction_gt_2)
        # voxels_tensor = reconstruction_gt_2.reshape(batch_size, 3, 20 * 752 * 752)
        # self.plot_voxel_point_cloud(voxels_tensor,
        #                             gt_bboxes_3d[0].tensor,
        #                             gt_labels_3d[0])
        # self.centerpoint_model_sparse.cp_plotter([points_fg[0]],
        #                                          gt_bboxes_3d[0].tensor,
        #                                          gt_labels_3d[0])
        # reconstruction_gt_4 = self.pcr_pool(self.pcr_pool(SparseConvTensor_4.dense()[:,:-1,:,:]))
        mask_loss, offset_loss = self.pcr_loss(
            reconstruction_gt_2, gen_offset_2, gen_mask_2,
            reconstruction_gt_4, gen_offset_4, gen_mask_4)
        pcr_losses['loss_mask'] = mask_loss
        pcr_losses['loss_offset'] = offset_loss

        return pcr_losses

    def pcr_loss(self, reconstruction_gt_2, gen_offset_2, gen_mask_2,
                       reconstruction_gt_4, gen_offset_4, gen_mask_4):

        grid_4, grid = None, None

        # Stage 1 PCR loss
        N, _, D, H, W = gen_offset_2.shape
        zs, ys, xs = torch.meshgrid([torch.arange(0,D), torch.arange(0, H), torch.arange(0, W)])
        ys = ys * (150.4 / H) - 75.2 + (150.4 / H) / 2
        xs = xs * (150.4 / W) - 75.2 + (150.4 / H) / 2
        zs = zs * (6 / D) - 2 + (6 / D) / 2
        grid = torch.cat([xs[None],ys[None],zs[None]],0)[None].repeat(N,1,1,1,1).to(gen_offset_2)

        # Stage 2 PCR loss
        N, _, D, H, W = reconstruction_gt_4.shape
        zs, ys, xs = torch.meshgrid([torch.arange(0,D), torch.arange(0, H), torch.arange(0, W)])
        ys = ys * (150.4 / H) - 75.2 + (150.4 / H) / 2
        xs = xs * (150.4 / W) - 75.2 + (150.4 / H) / 2
        zs = zs * (6 / D) - 2 + (6 / D) / 2
        grid_4 = torch.cat([xs[None],ys[None],zs[None]],0)[None].repeat(N,1,1,1,1).to(reconstruction_gt_4)

        mask_loss_2, offset_loss_2 = self.mask_offset_loss(gen_offset_2, gen_mask_2, reconstruction_gt_2, grid)
        mask_loss_4, offset_loss_4 = self.mask_offset_loss(gen_offset_4, gen_mask_4, reconstruction_gt_4, grid_4)
        mask_loss = mask_loss_2 + mask_loss_4
        comp_loss = offset_loss_2 + offset_loss_4

        return mask_loss, comp_loss

    def mask_offset_loss(self, gen_offset, gen_mask, gt, grid):

        gt_mask = gt.sum(1) != 0
        count_pos = gt_mask.sum()
        count_neg = (~gt_mask).sum()
        beta = count_neg / count_pos
        loss = F.binary_cross_entropy_with_logits(gen_mask[:,0], gt_mask.float(), pos_weight=beta) 

        grid = grid * gt_mask[:,None]
        gt = gt[:,:3] - grid
        gt_ind = gt != 0
        
        com_loss = F.l1_loss(gen_offset[gt_ind], gt[gt_ind])

        return loss, com_loss

    def simple_test(self, points, img_metas, img=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):
        """Test function without augmentation."""
        # self.plotter_test(points[0], gt_bboxes_3d[0][0], gt_labels_3d[0][0])
        img_feats, pts_feats, _, _, _ = self.centerpoint_model_sparse.extract_feat(
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

    def plotter(self, points, points_fg, gt_bboxes_3d, gt_labels_3d):
        """Function for performing sanity checks."""
        self.interp='None'
        self.ext=[-75.2, 75.2, -75.2, 75.2]
        self.orig='lower'

        plt.figure(figsize=(10*2,10))
        ax1 = plt.subplot(1,2,1)
        ax1.set_title(f'Dense Full Point Cloud', size=15, fontweight='bold')
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

        ax2 = plt.subplot(1,2,2)
        ax2.set_title(f'Complete FG Point Cloud', size=15, fontweight='bold')
        plt.scatter(
            points_fg[0][:,0].cpu(), points_fg[0][:,1].cpu(),
            s=0.001, alpha=0.5, c=points_fg[0][:,2].cpu(), cmap='viridis',
            vmin=torch.min(points[0][:,2].cpu()), vmax=torch.max(points[0][:,2].cpu())
        )
        self.plot_bboxes(gt_bboxes_3d[0], gt_labels_3d[0])
        plt.xlim(-75.2/3,75.2/3)
        plt.ylim(-75.2/3,75.2/3)
        ax2.get_xaxis().set_ticks([])
        ax2.get_yaxis().set_ticks([])

        plt.tight_layout()
        plt.savefig(f'in_domain_nusc_s2d_centerpoint_scatter33.png', dpi=100)
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
                                ec=cmap[label], alpha=float(score), fill=False,
                                transform=transform, linewidth=2)
            plt.gca().add_patch(rect)
        return None

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
        ax1.set_title(f'Dense FG + Sparse BG Point Cloud', size=15, fontweight='bold')
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
        plt.savefig(f'2d_voxel_point_cloud2.png', dpi=100)
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
        plt.savefig(f'3d_voxel_point_cloud2.png', dpi=100)
        plt.close()

        return None
