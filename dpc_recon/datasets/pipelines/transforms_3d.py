# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, box_np_ops)
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.datasets.pipelines import ObjectSample as _ObjectSample


@PIPELINES.register_module()
class ObjectSample2(_ObjectSample):
    """Sample GT objects to the data from both a sprase and dense db.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler2'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        static_ids = input_dict['static_ids']

        # change to float for blending operation
        points = input_dict['points']
        points_dense = input_dict['points_dense']
        points_fg = input_dict['points_fg']
        if self.sample_2d:
            img = input_dict['img']
            gt_bboxes_2d = input_dict['gt_bboxes']
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_points_dense = sampled_dict['points_dense']
            sampled_points_fg = sampled_dict['points_fg']
            sampled_gt_labels = sampled_dict['gt_labels_3d']
            sampled_static_ids = sampled_dict['static_ids']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            points_dense = self.remove_points_in_boxes(points_dense, sampled_gt_bboxes_3d)
            points_fg = self.remove_points_in_boxes(points_fg, sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])
            if len(sampled_points_dense) > 0:
                points_dense = points_dense.cat([sampled_points_dense, points_dense])
                points_fg = points_fg.cat([sampled_points_fg, points_fg])
            static_ids = np.concatenate([static_ids, sampled_static_ids], axis=0)

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.long)
        input_dict['points'] = points
        input_dict['points_dense'] = points_dense
        input_dict['points_fg'] = points_fg
        input_dict['static_ids'] = static_ids

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str


@PIPELINES.register_module(force=True)
class AddPointsTo(object):
    def __init__(self, points_from_names, points_to_names):
        if not isinstance(points_from_names, (list, tuple)):
            points_from_names = [points_from_names]
        if not isinstance(points_to_names, (list, tuple)):
            points_to_names = [points_to_names]
        self.points_from_names = points_from_names
        self.points_to_names = points_to_names
        assert len(self.points_from_names)==len(self.points_to_names)

    def __call__(self, input_dict):
        for points_from_name, points_to_name in zip(self.points_from_names, self.points_to_names):
            if points_from_name not in input_dict:
                continue
            if points_to_name not in input_dict:
                continue
            points_from = input_dict[points_from_name]
            points_to = input_dict[points_to_name]
            input_dict[points_to_name] = points_to.cat([points_from, points_to])
        return input_dict


@PIPELINES.register_module(force=True)
class DropPoints(object):
    def __init__(self, points_names):
        if not isinstance(points_names, (list, tuple)):
            points_names = [points_names]
        self.points_names = points_names

    def __call__(self, input_dict):
        for points_name in self.points_names:
            if points_name not in input_dict:
                continue
            del input_dict[points_name]
        return input_dict


@PIPELINES.register_module(force=True)
class RemoveObjectPoints(object):
    def __init__(self, points_name='points'):
        if not isinstance(points_name, (list, tuple)):
            points_name = [points_name]
        self.points_name = points_name

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        if 'gt_bboxes_3d' in input_dict:
            for name in self.points_name:
                points = input_dict[name]
                gt_bboxes_3d = input_dict['gt_bboxes_3d']
                points = self.remove_points_in_boxes(points, gt_bboxes_3d.tensor.numpy())

                ann_info = input_dict['ann_info']
                if 'sweeps' in ann_info and len(ann_info['sweeps']) > 0:
                    for i in range(len(input_dict['sweeps'])):
                        sweep_bboxes_3d = ann_info['sweeps'][i]['gt_bboxes_3d']
                        if len(sweep_bboxes_3d) > 0:
                            points = self.remove_points_in_boxes(points, sweep_bboxes_3d.tensor.numpy())

                input_dict[name] = points
        return input_dict
