# Copyright (c) OpenMMLab. All rights reserved.
from ...builder import MODULES
from .modules import Feat2FeatUnetSkipConnectionBlock
from mmgen.models.architectures.pix2pix import PatchDiscriminator as Pix2PixPatchDiscriminator
from mmgen.models.architectures.pix2pix import UnetGenerator as Pix2PixUnetGenerator


@MODULES.register_module()
class Feat2FeatUnetGenerator(Pix2PixUnetGenerator):
    """Construct the Unet-based generator from the innermost layer to the
    outermost layer, which is a recursive process.

    Args:
        in_channels (int): Number of channels in input images.
        out_channels (int): Number of channels in output images.
        num_down (int): Number of downsamplings in Unet. If `num_down` is 8,
            the image with size 256x256 will become 1x1 at the bottleneck.
            Default: 8.
        base_channels (int): Number of channels at the last conv layer.
            Default: 64.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='BN')`.
        use_dropout (bool): Whether to use dropout layers. Default: False.
        init_cfg (dict): Config dict for initialization.
            `type`: The name of our initialization method. Default: 'normal'.
            `gain`: Scaling factor for normal, xavier and orthogonal.
            Default: 0.02.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_down=8,
                 base_channels=64,
                 norm_cfg=dict(type='BN'),
                 use_dropout=False,
                 init_cfg=dict(type='normal', gain=0.02),
                 kernel_sizes=None,
                 padding_sizes=None,
                 striding_sizes=None):
        super().__init__(in_channels,
                         out_channels,
                         num_down,
                         base_channels,
                         norm_cfg,
                         use_dropout,
                         init_cfg)
        # We use norm layers in the unet generator.
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"

        # add the innermost layer
        unet_block = Feat2FeatUnetSkipConnectionBlock(
            base_channels * 8,
            base_channels * 8,
            in_channels=None,
            submodule=None,
            norm_cfg=norm_cfg,
            is_innermost=True,
            kernel_size=kernel_sizes[4],
            padding_size=padding_sizes[4],
            stride_size=striding_sizes[4])
        # add intermediate layers with base_channels * 8 filters
        for _ in range(num_down - 5):
            assert kernel_sizes is None or len(kernel_sizes)>5
            unet_block = Feat2FeatUnetSkipConnectionBlock(
                base_channels * 8,
                base_channels * 8,
                in_channels=None,
                submodule=unet_block,
                norm_cfg=norm_cfg,
                use_dropout=use_dropout)
        # gradually reduce the number of filters
        # from base_channels * 8 to base_channels
        unet_block = Feat2FeatUnetSkipConnectionBlock(
            base_channels * 4,
            base_channels * 8,
            in_channels=None,
            submodule=unet_block,
            norm_cfg=norm_cfg,
            kernel_size=kernel_sizes[3],
            padding_size=padding_sizes[3],
            stride_size=striding_sizes[3])
        unet_block = Feat2FeatUnetSkipConnectionBlock(
            base_channels * 2,
            base_channels * 4,
            in_channels=None,
            submodule=unet_block,
            norm_cfg=norm_cfg,
            kernel_size=kernel_sizes[2],
            padding_size=padding_sizes[2],
            stride_size=striding_sizes[2])
        unet_block = Feat2FeatUnetSkipConnectionBlock(
            base_channels,
            base_channels * 2,
            in_channels=None,
            submodule=unet_block,
            norm_cfg=norm_cfg,
            kernel_size=kernel_sizes[1],
            padding_size=padding_sizes[1],
            stride_size=striding_sizes[1])
        # add the outermost layer
        self.model = Feat2FeatUnetSkipConnectionBlock(
            out_channels,
            base_channels,
            in_channels=in_channels,
            submodule=unet_block,
            is_outermost=True,
            norm_cfg=norm_cfg,
            kernel_size=kernel_sizes[0],
            padding_size=padding_sizes[0],
            stride_size=striding_sizes[0])

        self.init_type = 'normal' if init_cfg is None else init_cfg.get(
            'type', 'normal')
        self.init_gain = 0.02 if init_cfg is None else init_cfg.get(
            'gain', 0.02)


@MODULES.register_module()
class Feat2FeatPatchDiscriminator(Pix2PixPatchDiscriminator):
    """A PatchGAN discriminator.

    Args:
        in_channels (int): Number of channels in input images.
        base_channels (int): Number of channels at the first conv layer.
            Default: 64.
        num_conv (int): Number of stacked intermediate convs (excluding input
            and output conv). Default: 3.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='BN')`.
        init_cfg (dict): Config dict for initialization.
            `type`: The name of our initialization method. Default: 'normal'.
            `gain`: Scaling factor for normal, xavier and orthogonal.
            Default: 0.02.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
