# Copyright (c) OpenMMLab. All rights reserved.
from ...builder import MODULES
from mmgen.models.architectures.cyclegan import ResnetGenerator as CycleGANResnetGenerator


@MODULES.register_module()
class CycleFeatGANResnetGenerator(CycleGANResnetGenerator):
    """Construct a Resnet-based generator that consists of residual blocks
    between a few downsampling/upsampling operations.
    
    Args:
        in_channels (int): Number of channels in input images.
        out_channels (int): Number of channels in output images.
        base_channels (int): Number of filters at the last conv layer.
            Default: 64.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='IN')`.
        use_dropout (bool): Whether to use dropout layers. Default: False.
        num_blocks (int): Number of residual blocks. Default: 9.
        padding_mode (str): The name of padding layer in conv layers:
            'reflect' | 'replicate' | 'zeros'. Default: 'reflect'.
        init_cfg (dict): Config dict for initialization.
            `type`: The name of our initialization method. Default: 'normal'.
            `gain`: Scaling factor for normal, xavier and orthogonal.
            Default: 0.02.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
