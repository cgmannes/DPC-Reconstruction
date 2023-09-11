# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from .utils import Empty, GroupNorm, Sequential
from mmcv.runner import auto_fp16
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.modules.conv import Conv2d

from mmdet.models import NECKS
from mmdet3d.models.necks import SECONDFPN


@NECKS.register_module()
class S2DSECONDFPN(SECONDFPN):
    """Modified/extended FPN originally used in SECOND/PointPillars/PartA2/MVXNet
    with an additional Resnet module and fusion technique from Sparse2Dense.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self, training, pcr, num_input_features=256, *args, **kwargs):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super().__init__(*args, **kwargs)
        self.training = training
        self.pcr = pcr

        # S2D module
        self.encoder_1 = Sequential( # 94,94,256
            Conv2d(num_input_features,256,2,2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.encoder_2 = Sequential( # 47, 47, 512
            Conv2d(256,256,3,2,1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
        )
        self.convnext_block_1 = Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.LayerNorm([256,47,47], eps=1e-6),
            nn.Conv2d(256,256*4,1,1,0),
            nn.GELU(),
            nn.Conv2d(256*4,256,1,1,0),
        )
        self.convnext_block_2 = Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.LayerNorm([256,47,47], eps=1e-6),
            nn.Conv2d(256,256*4,1,1,0),
            nn.GELU(),
            nn.Conv2d(256*4,256,1,1,0),
        )
        self.convnext_block_3 = Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.LayerNorm([256,47,47], eps=1e-6),
            nn.Conv2d(256,256*4,1,1,0),
            nn.GELU(),
            nn.Conv2d(256*4,256,1,1,0),
        )
        self.decoder_1 = Sequential( # 94,94,256
            nn.ConvTranspose2d(256,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.GELU(),  
        )
        self.decoder_2 = Sequential( # 188,188,256
            nn.Conv2d(512,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256,num_input_features,4,2,1),
            nn.BatchNorm2d(num_input_features),
            nn.GELU(),
        )
        self.fusion_sparse = Sequential(
            nn.Conv2d(num_input_features,num_input_features,1,1,0),
            nn.BatchNorm2d(num_input_features),
            nn.GELU(),
        )
        self.fusion_dense = Sequential(
            nn.Conv2d(num_input_features,num_input_features,1,1,0),
            nn.BatchNorm2d(num_input_features),
            nn.GELU(),
        )
        self.out_conv = Sequential(
            nn.Conv2d(num_input_features,640,1,1,0),
            nn.BatchNorm2d(640),
            nn.GELU(),
        )

        # PCR module
        if self.pcr:
            self.generator_1 = Sequential( # N,128,5,188,188
                nn.Conv3d(128,32,1,1,0),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.ConvTranspose3d(32,32,4,2,1), # N,32,10,376,376
                nn.BatchNorm3d(32),
                nn.ReLU(),
            )
            self.gen_out_4 = Sequential(
                nn.Conv3d(32,3,1,1,0),
            )
            self.gen_mask_4 = Sequential(
                nn.Conv3d(32,1,1,1,0),
            )
            self.generator_2 = Sequential(
                nn.Conv3d(32,16,1,1,0),
                nn.BatchNorm3d(16),
                nn.ReLU(),
                nn.ConvTranspose3d(16,3,4,2,1), # N,16,20,752,752
                nn.BatchNorm3d(3),
                nn.ReLU(),
            )
            self.gen_out_2 = Sequential(
                nn.Conv3d(3,3,1,1,0),
            )
            self.gen_mask_2 = Sequential(
                nn.Conv3d(3,1,1,1,0),
            )

    @auto_fp16()
    def forward_s2d(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            z, y (torch.Tensor): 4D Tensor in (N, C, H, W) shape.
        """
        # assert len(x) == len(self.in_channels)

        # S2D Module
        y_1 = self.encoder_1(x)                                 # 94
        y_2 = self.encoder_2(y_1)                               # 47
        att = self.convnext_block_1(y_2) + y_2                  # 47
        att = self.convnext_block_2(att) + att                  # 47
        att = F.gelu(self.convnext_block_3(att) + att)          # 47
        y_3 = torch.cat([self.decoder_1(att) , y_1],1)          # 94
        F_S_b = self.decoder_2(y_3)                               # 188
        F_S_a = self.fusion_dense(F_S_b) + self.fusion_sparse(x)  # 188

        return F_S_a, F_S_b

    @auto_fp16()
    def forward_pcr(self, F_S_b, N=None, _=None, H=None, W=None):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            z, y (torch.Tensor): 4D Tensor in (N, C, H, W) shape.
        """
        # assert len(x) == len(self.in_channels)

        # PCR Module
        if self.training and self.pcr:
            gen = self.out_conv(F_S_b)
            gen = gen.view(N,128,5,H,W)
            gen = self.generator_1(gen)
            gen_offset_4 = self.gen_out_4(gen)
            gen_mask_4 = self.gen_mask_4(gen)
            gen = self.generator_2(gen)
            gen_mask_2 = self.gen_mask_2(gen)
            gen_offset_2 = self.gen_out_2(gen)
        else:
            gen_offset_2, gen_mask_2, gen_offset_4, gen_mask_4 = None, None, None, None

        return gen_offset_2, gen_mask_2, gen_offset_4, gen_mask_4

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        # assert len(x) == len(self.in_channels)

        # SECOND Neck
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]
