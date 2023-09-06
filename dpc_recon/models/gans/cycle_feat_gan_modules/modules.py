# Copyright (c) OpenMMLab. All rights reserved.
from ...builder import MODULES
from mmgen.models.architectures.cyclegan import ResidualBlockWithDropout as CycleGANResidualBlockWithDropout


@MODULES.register_module()
class CycleFeatGANResidualBlockWithDropout(CycleGANResidualBlockWithDropout):
    """Define a Residual Block with dropout layers.

    Ref:
      Deep Residual Learning for Image Recognition
      A residual block is a conv block with skip connections. A dropout layer is
      added between two common conv modules.
    
    Args:
        channels (int): Number of channels in the conv layer.
        padding_mode (str): The name of padding layer:
            'reflect' | 'replicate' | 'zeros'.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='IN')`.
        use_dropout (bool): Whether to use dropout layers. Default: True.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
