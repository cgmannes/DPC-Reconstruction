# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.models.builder import *
from mmcv.utils import Registry, build_from_cfg
from mmgen.models.builder import MODULES
from mmgen.models.builder import MODELS as GAN_MODELS


def build_gan_model(cfg):
    """Build gan translation model."""
    return GAN_MODELS.build(cfg)

def build_module(cfg, default_args=None):
    """Build a module."""
    # print(cfg)
    # for key, value in sorted(MODULES.module_dict.items()):
    #     print(f'{key} and {value}')
    # exit()
    return build_from_cfg(cfg, MODULES, default_args)
