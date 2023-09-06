# Copyright (c) stevewongv/Sparse2Dense. All rights reserved.
# https://github.com/stevewongv/Sparse2Dense
import torch
import torch.nn.functional as F

from mmgen.models.builder import MODULES


@MODULES.register_module()
class Sparse2DensePCRLoss(object):
    """PCR loss based on Sparse2Dense: Learning to Densify 3D Features. This
    class is used to perform the reconstruction task
    https://arxiv.org/pdf/2211.13067.pdf

    Args:
        beta_a (int): Weight of the non-zero elements in the main branch of
            the Sparse2Dense framework.
        gamma_a (int): Weight of the zero elements in the main branch of the
            Sparse2Dense framework.
        beta_b (int, optional): Weight of the non-zero elements in the point
            cloud reconstructionn branch of the Sparse2Dense framework.
        gamma_b (int, optional): Weight of the zero elements in the point cloud
            reconstruction branch of the Sparse2Dense framework.
    """
    def __init__(self):
        super().__init__()

    def forward(self, reconstruction_gt_2, gen_offset_2, gen_mask_2,
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