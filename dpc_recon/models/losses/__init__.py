# Copyright (c) OpenMMLab. All rights reserved.
from .mse_pixelwise_loss import S2DMSELoss
from .sparse_pixelwise_loss import SparseL1Loss
from .sparse2dense_dis_loss import Sparse2DenseDisLoss
from .sparse2dense_mse_loss import Sparse2DenseMSELoss
from .sparse2dense_pcr_loss import Sparse2DensePCRLoss

__all__ = [
    'S2DMSELoss', 'SparseL1Loss', 'Sparse2DenseDisLoss', 'Sparse2DenseMSELoss', 'Sparse2DensePCRLoss'
]