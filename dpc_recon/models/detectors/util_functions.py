# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model


def init_model(config, checkpoint):
    """Initialize a model from config file for a 3D detector.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Device to use.

    Returns:
        nn.Module: The constructed detector.
    """
    config.pretrained = None
    model = build_model(config)
    checkpoint = load_checkpoint(model, checkpoint)

    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = config.class_names
    if 'PALETTE' in checkpoint['meta']:  # 3D Segmentor
        model.PALETTE = checkpoint['meta']['PALETTE']
    
    # Save the config in the model for convenience.
    model.cfg = config

    return model

def feat2feat_visualization(feat_maps, bboxes, source_domain, target_domain, reducer=None):
    feat2feat_results = dict()

    num_channels = feat_maps[f'real_{source_domain}'].shape[1]
    height = feat_maps[f'real_{source_domain}'].shape[2]
    width = feat_maps[f'real_{source_domain}'].shape[3]

    if reducer is not None:
        feat2feat_results[f'real_{source_domain}'] = torch.t(torch.tensor(
            reducer.fit_transform(
                feat_maps[f'real_{source_domain}'].clone().detach().cpu()\
                    .permute(0, 3, 2, 1)[0].reshape(-1, num_channels)
            ).reshape(width, height)
        ))
        feat2feat_results[f'fake_{target_domain}'] = torch.t(torch.tensor(
            reducer.fit_transform(
                feat_maps[f'fake_{target_domain}'].clone().detach().cpu()\
                    .permute(0, 3, 2, 1)[0].reshape(-1, num_channels)
            ).reshape(width, height)
        ))
        feat2feat_results[f'real_{target_domain}'] = torch.t(torch.tensor(
            reducer.fit_transform(
                feat_maps[f'real_{target_domain}'].clone().detach().cpu()\
                    .permute(0, 3, 2, 1)[0].reshape(-1, num_channels)
            ).reshape(width, height)
        ))
    else:
        feat2feat_results[f'full_real_{source_domain}'] = \
            feat_maps[f'real_{source_domain}'].clone().detach().cpu()[0]
        feat2feat_results[f'real_{source_domain}'] = torch.mean(
            feat_maps[f'real_{source_domain}'].clone().detach().cpu(), dim=1)[0]
        
        feat2feat_results[f'full_fake_{target_domain}'] = \
            feat_maps[f'fake_{target_domain}'].clone().detach().cpu()[0]
        feat2feat_results[f'fake_{target_domain}'] = torch.mean(
            feat_maps[f'fake_{target_domain}'].clone().detach().cpu(), dim=1)[0]
        
        feat2feat_results[f'full_real_{target_domain}'] = \
            feat_maps[f'real_{target_domain}'].clone().detach().cpu()[0]
        feat2feat_results[f'real_{target_domain}'] = torch.mean(
            feat_maps[f'real_{target_domain}'].clone().detach().cpu(), dim=1)[0]
        
        feat2feat_results[f'fft_fake_{target_domain}'] = np.fft.fftshift(np.fft.fft2(
            feat2feat_results[f'fake_{target_domain}']))
        feat2feat_results[f'fft_real_{target_domain}'] = np.fft.fftshift(np.fft.fft2(
            feat2feat_results[f'real_{target_domain}']))
    feat2feat_results[f'heatmap_masks'] = feat_maps['heatmap_masks'][0].squeeze(0).clone().detach().cpu()
    feat2feat_results[f'gt_bboxes_3d'] = bboxes[0]

    return feat2feat_results