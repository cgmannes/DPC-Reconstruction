# Copyright (c) stevewongv/Sparse2Dense. All rights reserved.
# https://github.com/stevewongv/Sparse2Dense
import torch
import torch.nn.functional as F

from mmgen.models.builder import MODULES


@MODULES.register_module()
class Sparse2DenseMSELoss(object):
    """MSE loss based on Sparse2Dense: Learning to Densify 3D Features. This
    class is used to perform the +distillation ablation study
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
    def __init__(self, beta_a, gamma_a, beta_b=None, gamma_b=None):
        super().__init__()
        self.beta_a = beta_a
        self.gamma_a = gamma_a
        self.beta_b = beta_b
        self.gamma_b = gamma_b

    def forward(self, F_S_a, F_D_a, F_S_b=None, F_D_b=None, ignore_mask=None):
        assert (F_S_b is None and F_D_b is None) or \
            (F_S_b is not None and F_D_b is not None)

        if ignore_mask is not None:
            inds = ignore_mask > 0
            inds_F_D_a = F_D_a > 0
            mask = torch.logical_and(inds, inds_F_D_a)
            sparse2dense_loss = F.mse_loss((F_S_a)[mask],(F_D_a)[mask]) * self.beta_a
            sparse2dense_loss += F.mse_loss((F_S_a)[~mask],(F_D_a)[~mask]) * self.gamma_a

            if F_S_b is not None and F_D_b is not None:
                inds_F_D_b = F_D_b > 0
                if inds.shape[0] != inds_F_D_b.shape[0]:
                    return sparse2dense_loss
                mask = torch.logical_and(inds, inds_F_D_b)
                sparse2dense_loss += F.mse_loss((F_S_b)[mask],(F_D_b)[mask]) * self.beta_b
                sparse2dense_loss += F.mse_loss((F_S_b)[~mask],(F_D_b)[~mask]) * self.gamma_b
        else:
            mask = F_D_a > 0
            sparse2dense_loss = F.mse_loss((F_S_a)[mask],(F_D_a)[mask]) * self.beta_a
            sparse2dense_loss += F.mse_loss((F_S_a)[~mask],(F_D_a)[~mask]) * self.gamma_a

            if F_S_b is not None and F_D_b is not None:
                mask = F_D_b > 0
                sparse2dense_loss += F.mse_loss((F_S_b)[mask],(F_D_b)[mask]) * self.beta_b
                sparse2dense_loss += F.mse_loss((F_S_b)[~mask],(F_D_b)[~mask]) * self.gamma_b

        return sparse2dense_loss