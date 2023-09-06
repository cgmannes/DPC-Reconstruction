# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import os
import torch

from mmdet.datasets.builder import PIPELINES
from mmdet3d.core import LiDARInstance3DBoxes, LiDARPoints
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.datasets.pipelines import LoadAnnotations3D as _LoadAnnotations3D
from mmdet3d.datasets.pipelines import LoadPointsFromFile as _LoadPointsFromFile
from mmdet3d.datasets.pipelines import LoadPointsFromMultiSweeps as _LoadPointsFromMultiSweeps


@PIPELINES.register_module(force=True)
class LoadAnnotations3D(_LoadAnnotations3D):
    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        if 'ann_info' in results:
            results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
            if len(results['gt_bboxes_3d'])!=0:
                results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        if 'ann_info' in results:
            results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
            if 'gt_scores_3d' in results['ann_info']:
                results['gt_scores_3d'] = results['ann_info']['gt_scores_3d']
            if 'gt_ignores_3d' in results['ann_info']:
                results['gt_ignores_3d'] = results['ann_info']['gt_ignores_3d']
        return results


@PIPELINES.register_module(force=True)
class LoadPointsFromMultiSweeps(_LoadPointsFromMultiSweeps):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2, 3, 4],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk'),
                 sweeps_num=20,
                 pad_empty_sweeps=True,
                 remove_close=True,
                 test_mode=False):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

        self.sweeps_num = sweeps_num
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        if self.sweeps_num == 0:
            points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        if self.sweeps_num==0:
            return results

        points = results['points'].tensor
        points[:, 4] = 0
        if self.remove_close:
            points = self._remove_close(points.numpy())
            points = torch.tensor(points).to(dtype=torch.float32)
        cur_pose = torch.tensor(results['pose']).to(dtype=torch.float32)
        points[:,:3] = points[:,:3] @ cur_pose[:3,:3].T
        points[:,:3] += cur_pose[:3,3]
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    points = self._remove_close(points.tensor.numpy())
                    points = points_class(
                        points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
                    sweep_points_list.append(points)
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                pts_filename = sweep['velodyne_path']
                points_sweep = self._load_points(f'data/waymo/kitti_format/{pts_filename}')
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                points_sweep = torch.tensor(points_sweep).to(dtype=torch.float32)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep.numpy())
                    points_sweep = torch.tensor(points_sweep).to(dtype=torch.float32)
                sweep_ts = sweep['timestamp'] / 1e6
                pose = torch.tensor(sweep['pose']).to(dtype=torch.float32)
                points_sweep[:,:3] = points_sweep[:,:3] @ pose[:3,:3].T
                points_sweep[:,:3] += pose[:3,3]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points_class(
                    points_sweep, points_dim=points_sweep.shape[-1], attribute_dims=attribute_dims)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points.tensor.to(dtype=torch.float32)
        points[:,:3] -= cur_pose[:3,3]
        points[:,:3] = points[:,:3] @ cur_pose[:3,:3]
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points[:, self.use_dim]

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str
