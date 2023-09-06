# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D


def gaussian_2d(shape, sigma=1):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float, optional): Sigma to generate gaussian map.
            Defaults to 1.
    
    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        K (int, optional): Multiple of masked_gaussian. Defaults to 1.
    
    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.5):
    """Get radius of gaussian.

    Args:
        det_size (tuple[torch.Tensor]): Size of the detection result.
        min_overlap (float, optional): Gaussian_overlap. Defaults to 0.5.
    
    Returns:
        torch.Tensor: Computed radius.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def get_ellip_gaussian_2D(heatmap, center, radius_x, radius_y,
                          yaw, k=1, gt_bboxes_3d=None, obj_num=0):
    """Modified function to generate 2D ellipse gaussian heatmap.

    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius_x (int): X-axis radius of gaussian kernel.
        radius_y (int): Y-axis radius of gaussian kernel.
        yaw (float): The yaw rotation of the object in degrees.
        k (int, optional): Coefficient of gaussian kernel. For domain
                           adaptation, k is the L1 weight parameter.
                           Default: 1.
        gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
        obj_num (int): Integer for generating plots of the masking process.
    
    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    diameter_x, diameter_y = 2 * radius_x + 1, 2 * radius_y + 1
    gaussian_kernel = ellip_gaussian2D((torch.max(torch.tensor([radius_x, radius_y])),
                                        torch.max(torch.tensor([radius_x, radius_y]))),
                                        sigma_x=diameter_x / 13,
                                        sigma_y=diameter_y / 13,
                                        dtype=heatmap.dtype,
                                        device=heatmap.device)
    # plot_mask(gaussian_kernel, obj_num, 'a_gaussian kernel')

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    max_radius = int(torch.max(torch.tensor([radius_x, radius_y])))
    left, right = int(min(x, max_radius)), int(min(width - x, max_radius + 1))
    top, bottom = int(min(y, max_radius)), int(min(height - y, max_radius + 1))

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[max_radius - top:max_radius + bottom,
                                      max_radius - left:max_radius + right]
    masked_gaussian[masked_gaussian>0] = 1

    # plot_mask(masked_gaussian, obj_num, 'b_masked_gaussian_init')
    masked_gaussian = TF.rotate(masked_gaussian.unsqueeze(0), yaw).squeeze(0)
    # plot_mask(masked_gaussian, obj_num, 'c_masked_gaussian_rotated')

    out_heatmap = heatmap
    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right])

    # object_mask = torch.zeros(height, width)
    # object_mask[y - top:y + bottom, x - left:x + right] = 1

    # interp='None'
    # ext=[-75.2, 75.2, -75.2, 75.2]
    # orig='lower'

    # plt.figure(figsize=(15,15))
    # ax = plt.subplot(1,1,1)
    # ax.set_title(f'HeatMap Mask', size=20, fontweight='bold')
    # plt.imshow(out_heatmap.squeeze().cpu().numpy(), interpolation=interp, extent=ext, origin=orig)
    # plt.colorbar()
    # for box in gt_bboxes_3d.cpu().numpy():
    #     plt.gca().add_patch(Rectangle(
    #         box[:2]-box[3:5]/2, box[3], box[4],
    #         lw=1, ec='tab:orange', fc='tab:orange', alpha=0.25,
    #         transform=Affine2D().rotate_around(box[0], box[1], -box[6]) + 
    #             plt.gca().transData
    #     ))

    # plt.tight_layout()
    # plt.savefig(f'{obj_num}d_ellip_gaussian_2D_obj_mask.png')
    # plt.close()

    return out_heatmap


def ellip_gaussian2D(radius,
                     sigma_x,
                     sigma_y,
                     dtype=torch.float32,
                     device='cpu'):
    """Generate 2D ellipse gaussian kernel.

    Args:
        radius (tuple(int)): Ellipse radius (radius_x, radius_y) of gaussian
            kernel.
        sigma_x (int): X-axis sigma of gaussian function.
        sigma_y (int): Y-axis sigma of gaussian function.
        dtype (torch.dtype, optional): Dtype of gaussian tensor.
            Default: torch.float32.
        device (str, optional): Device of gaussian tensor.
            Default: 'cpu'.
    
    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius_y + 1) * (2 * radius_x + 1)`` shape.
    """
    x = torch.arange(
        -radius[0], radius[0] + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius[1], radius[1] + 1, dtype=dtype, device=device).view(-1, 1)

    h = (-(x * x) / (2 * sigma_x * sigma_x) - (y * y) /
         (2 * sigma_y * sigma_y)).exp()
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0

    return h


def plot_mask(mask, obj_num, title):
    '''
    Plotting function for visualizing the masking process.
    '''
    interp='None'
    orig='lower'

    plt.figure(figsize=(15,15))
    ax = plt.subplot(1,1,1)
    ax.set_title(f'Plot {title} in Masking Process', size=20, fontweight='bold')
    plt.imshow(mask.cpu().numpy(), interpolation=interp, origin=orig)
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f'{obj_num}{title}.png')
    plt.close()
    
    return None