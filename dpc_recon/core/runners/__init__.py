# Copyright (c) OpenMMLab. All rights reserved.
from .mmdet3d_mmgen_dynamic_iterbased_runner import MMDet3dMMGenDynamicIterBasedRunner
from .mmdet3d_mmgen_epoch_based_runner import MMDet3dMMGenEpochBasedRunner
from mmgen.core.runners import dynamic_iterbased_runner

__all__ = [
    'MMDet3dMMGenDynamicIterBasedRunner', 'MMDet3dMMGenEpochBasedRunner',
    'dynamic_iterbased_runner'
]
