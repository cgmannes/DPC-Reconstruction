# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import os
import torch

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomFlip as _RandomFlip
from mmdet3d.core import LiDARInstance3DBoxes, LiDARPoints
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.datasets.pipelines import GlobalRotScaleTrans as _GlobalRotScaleTrans
from mmdet3d.datasets.pipelines import LoadAnnotations3D as _LoadAnnotations3D
from mmdet3d.datasets.pipelines import LoadPointsFromFile as _LoadPointsFromFile
from mmdet3d.datasets.pipelines import LoadPointsFromMultiSweeps as _LoadPointsFromMultiSweeps
from mmdet3d.datasets.pipelines import RandomFlip3D as _RandomFlip3D
from simuda.datasets.pipelines.transforms_3d import RemoveGroundPoints as _RemoveGroundPoints


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
                 dataset_type,
                 load_dim=6,
                 use_dim=[0, 1, 2, 3, 4],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk'),
                 sweeps_num=0,
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
        assert dataset_type.lower() in ['waymo', 'nuscenes'], \
            f'The dataset_type must be waymo or nuscenes'

        self.coord_type = coord_type
        self.dataset_type = dataset_type.lower()
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
        if self.dataset_type in ['waymo']:
            cur_pose = torch.tensor(results['pose']).to(dtype=torch.float32)
            points[:,:3] = points[:,:3] @ cur_pose[:3,:3].T
            points[:,:3] += cur_pose[:3,3]
        elif self.dataset_type in ['nuscenes']:
            points[:, :3] = points[:, :3] @ results['lidar2ego_rotation'].T
            points[:, :3] += results['lidar2ego_translation']
            points[:, :3] = points[:, :3] @ results['ego2global_rotation'].T
            points[:, :3] += results['ego2global_translation']
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
                if self.dataset_type in ['waymo']:
                    pts_filename = sweep['velodyne_path']
                    points_sweep = self._load_points(f'data/waymo/kitti_format/{pts_filename}')
                elif self.dataset_type in ['nuscenes']:
                    points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                points_sweep = torch.tensor(points_sweep).to(dtype=torch.float32)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep.numpy())
                    points_sweep = torch.tensor(points_sweep).to(dtype=torch.float32)
                sweep_ts = sweep['timestamp'] / 1e6
                if self.dataset_type in ['waymo']:
                    pose = torch.tensor(sweep['pose']).to(dtype=torch.float32)
                    points_sweep[:,:3] = points_sweep[:,:3] @ pose[:3,:3].T
                    points_sweep[:,:3] += pose[:3,3]
                elif self.dataset_type in ['nuscenes']:
                    points_sweep[:, :3] = points_sweep[:, :3] @ sweep['lidar2ego_rotation'].T
                    points_sweep[:, :3] += sweep['lidar2ego_translation']
                    points_sweep[:, :3] = points_sweep[:, :3] @ sweep['ego2global_rotation'].T
                    points_sweep[:, :3] += sweep['ego2global_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points_class(
                    points_sweep, points_dim=points_sweep.shape[-1], attribute_dims=attribute_dims)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points.tensor.to(dtype=torch.float32)
        if self.dataset_type in ['waymo']:
            points[:,:3] -= cur_pose[:3,3]
            points[:,:3] = points[:,:3] @ cur_pose[:3,:3]
        elif self.dataset_type in ['nuscenes']:
            points[:, :3] -= results['ego2global_translation']
            points[:, :3] = points[:, :3] @ results['ego2global_rotation']
            points[:, :3] -= results['lidar2ego_translation']
            points[:, :3] = points[:, :3] @ results['lidar2ego_rotation']
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

@PIPELINES.register_module(force=True)
class RemoveGroundPoints(_RemoveGroundPoints):
    def __init__(self, sweeps_num=0, points_name='points', voxel_size=[2, 2],
                 z_quantile=0.1, z_threshold=0.3):
        import open3d.ml.torch as ml3d
        if not isinstance(points_name, (list, tuple)):
            points_name = [points_name]
        self.sweeps_num = sweeps_num
        self.points_name = points_name
        self.voxel_size = voxel_size
        self.z_quantile = z_quantile
        self.z_threshold = z_threshold
        self.voxelize = ml3d.ops.voxelize

    def __call__(self, input_dict):
        if self.sweeps_num==0:
            return input_dict

        for name in self.points_name:
            points = input_dict[name]
            points_t = points.tensor
            height = points_t[:,2].max() - points_t[:,2].min()
            result = self.voxelize(
                points_t[:,:3],
                torch.tensor([0, points.shape[0]]),
                torch.tensor([*self.voxel_size, height], dtype=torch.float32),
                points_t[:,:3].min(0).values,
                points_t[:,:3].max(0).values)
            indices = []
            for i in range(len(result.voxel_point_row_splits)-1):
                begin, end = result.voxel_point_row_splits[i:i+2]
                indices_i = result.voxel_point_indices[begin:end]
                points_i = points_t[indices_i]
                z_min = points_i[:,2].quantile(self.z_quantile)
                indices.append(indices_i[points_i[:,2] > z_min + self.z_threshold])
            input_dict[name] = points[torch.cat(indices)]
        return input_dict

@PIPELINES.register_module(force=True)
class GlobalRotScaleTrans(_GlobalRotScaleTrans):
    """Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of translation
            noise. This applies random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        input_dict['points'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor
        for key in input_dict['bbox3d_fields']:
            input_dict[key].translate(trans_factor)

    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        # if no bbox in input_dict, only rotate points
        if len(input_dict['bbox3d_fields']) == 0:
            rot_mat_T = input_dict['points'].rotate(noise_rotation)
            input_dict['pcd_rotation'] = rot_mat_T
            return

        # rotate points with bboxes
        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:
                points, rot_mat_T = input_dict[key].rotate(
                    noise_rotation, input_dict['points'])
                input_dict['points'] = points
                input_dict['pcd_rotation'] = rot_mat_T

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        points = input_dict['points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['points'] = points

        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)

    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated \
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and \
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str

@PIPELINES.register_module(force=True)
class RandomFlip3D(_RandomFlip):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.5,
                 flip_ratio_bev_vertical=0.5,
                 **kwargs):
        super(RandomFlip3D, self).__init__(
            flip_ratio=flip_ratio_bev_horizontal, **kwargs)
        self.sync_2d = sync_2d
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        # for semantic segmentation task, only points will be flipped.
        if 'bbox3d_fields' not in input_dict:
            input_dict['points'].flip(direction)
            return
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1
        for key in input_dict['bbox3d_fields']:
            if 'points' in input_dict:
                input_dict['points'] = input_dict[key].flip(
                    direction, points=input_dict['points'])
            else:
                input_dict[key].flip(direction)
        if 'centers2d' in input_dict:
            assert self.sync_2d is True and direction == 'horizontal', \
                'Only support sync_2d=True and horizontal flip with images'
            w = input_dict['ori_shape'][1]
            input_dict['centers2d'][..., 0] = \
                w - input_dict['centers2d'][..., 0]
            # need to modify the horizontal position of camera center
            # along u-axis in the image (flip like centers2d)
            # ['cam2img'][0][2] = c_u
            # see more details and examples at
            # https://github.com/open-mmlab/mmdetection3d/pull/744
            input_dict['cam2img'][0][2] = w - input_dict['cam2img'][0][2]

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and \
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction', \
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added \
                into result dict.
        """
        # flip 2D image and its annotations
        super(RandomFlip3D, self).__call__(input_dict)

        if self.sync_2d:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d},'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str
