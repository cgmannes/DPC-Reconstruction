import copy
import mmcv
import numpy as np
import os

from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.bbox import box_np_ops
from mmdet3d.datasets.pipelines import data_augment_utils
from mmdet3d.datasets.pipelines import DataBaseSampler as _DataBaseSampler
from mmdet3d.datasets.builder import OBJECTSAMPLERS


class BatchSampler2:
    """Class for sampling specific category of ground truths for
    2 synced db infos.

    Args:
        sample_list (list[dict]): List of samples.
        name (str | None): The category of samples. Default: None.
        epoch (int | None): Sampling epoch. Default: None.
        shuffle (bool): Whether to shuffle indices. Default: False.
        drop_reminder (bool): Drop reminder. Default: False.
    """

    def __init__(self,
                 sampled_list,
                 name=None,
                 seed=42,
                 epoch=None,
                 shuffle=True,
                 drop_reminder=False):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        self.seed = seed
        if shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        """Sample specific number of ground truths.

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


@OBJECTSAMPLERS.register_module(force=True)
class DataBaseSampler2(_DataBaseSampler):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str]): List of classes. Default: None.
        points_loader(dict): Config of points loader. Default: dict(
            type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
    """

    def __init__(self,
                 info_path1,
                 info_path2,
                 data_root1,
                 data_root2,
                 rate,
                 prepare,
                 sample_groups,
                 classes=None,
                 points_loader1=dict(
                     type='LoadPointsFromFile',
                     coord_type='LIDAR',
                     load_dim=4,
                     use_dim=[0, 1, 2, 3]),
                 points_loader2=dict(
                     type='LoadPointsFromFile',
                     coord_type='LIDAR',
                     load_dim=4,
                     use_dim=[0, 1, 2, 3])):
        self.data_root1 = data_root1
        self.data_root2 = data_root2
        self.info_path1 = info_path1
        self.info_path2 = info_path2
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.label2cat = {i: name for i, name in enumerate(classes)}
        self.points_loader1 = mmcv.build_from_cfg(points_loader1, PIPELINES)
        self.points_loader2 = mmcv.build_from_cfg(points_loader2, PIPELINES)

        db_infos1 = mmcv.load(info_path1)
        db_infos2 = mmcv.load(info_path2)

        # filter database infos
        from mmdet3d.utils import get_root_logger
        logger = get_root_logger()
        for k, v in db_infos1.items():
            logger.info(f'load {len(v)} {k} database infos')
        for k, v in db_infos2.items():
            logger.info(f'load {len(v)} {k} database infos2')
        for prep_func, val in prepare.items():
            db_infos1, db_infos2 = getattr(self, prep_func)(db_infos1, db_infos2, val)
        logger.info('After filter database:')
        for k, v in db_infos1.items():
            logger.info(f'load {len(v)} {k} database infos')
        for k, v in db_infos2.items():
            logger.info(f'load {len(v)} {k} database infos2')

        self.db_infos1 = db_infos1
        self.db_infos2 = db_infos2

        # load sample groups
        # TODO: more elegant way to load sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.group_db_infos1 = self.db_infos1  # just use db_infos
        self.group_db_infos2 = self.db_infos2  # just use db_infos2
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict1 = {}
        for k, v in self.group_db_infos1.items():
            self.sampler_dict1[k] = BatchSampler2(v, k, shuffle=True)
        self.sampler_dict2 = {}
        for k, v in self.group_db_infos2.items():
            self.sampler_dict2[k] = BatchSampler2(v, k, shuffle=True)
        # TODO: No group_sampling currently

    @staticmethod
    def filter_by_difficulty(db_infos1, db_infos2, removed_difficulty):
        """Filter ground truths by difficulties.

        Args:
            db_infos1 (dict): Info of sparse groundtruth database.
            db_infos2 (dict): Info of dense groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after filtering.
        """
        new_db_infos1 = {}
        new_db_infos2 = {}
        for (key1, dinfos1),(key2, dinfos2) in zip(db_infos1.items(), db_infos2.items()):
            new_db_infos1[key1] = [
                info for info in dinfos1
                if info['difficulty'] not in removed_difficulty
            ]
            new_db_infos2[key2] = [
                info for info in dinfos2
                if info['difficulty'] not in removed_difficulty
            ]
        return new_db_infos1, new_db_infos2

    @staticmethod
    def filter_by_min_points(db_infos1, db_infos2, min_gt_points_dict):
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos1 (dict): Info of sparse groundtruth database.
            db_infos2 (dict): Info of dense groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos1, filtered_infos2 = [], []
                for info1, info2 in zip(db_infos1[name], db_infos2[name]):
                    if info2['num_points_in_gt'] >= min_num:
                        filtered_infos1.append(info1)
                        filtered_infos2.append(info2)
                db_infos1[name] = filtered_infos1
                db_infos2[name] = filtered_infos2
        return db_infos1, db_infos2

    def sample_all(self, gt_bboxes, gt_labels, img=None):
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels \
                    of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): \
                    sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            class_label = self.cat2label[class_name]
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(max_sample_num -
                              np.sum([n == class_label for n in gt_labels]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled1, sampled2 = [], []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes

        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls1, sampled_cls2 = self.sample_class_v2(class_name, sampled_num,
                                                                  avoid_coll_boxes)

                sampled1 += sampled_cls1
                sampled2 += sampled_cls2
                if len(sampled_cls1) > 0:
                    if len(sampled_cls1) == 1:
                        sampled_gt_box = sampled_cls1[0]['box3d_lidar'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls1], axis=0)

                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box[:,:avoid_coll_boxes.shape[1]]], axis=0)

        ret = None
        if len(sampled1) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            # center = sampled_gt_bboxes[:, 0:3]

            # num_sampled = len(sampled)
            s_points_list1, s_points_list2 = [], []
            group_ids, static_ids = [], []
            count = 0
            for info1, info2 in zip(sampled1, sampled2):
                file_path1 = os.path.join(
                    self.data_root1,
                    info1['path']) if self.data_root1 else info1['path']
                dir_name = info2['path'].split('/',1)[0]
                file_name = info1['path'].split('/',1)[1]
                file_path2 = os.path.join(
                    self.data_root2, dir_name, file_name)
                results1 = dict(pts_filename=file_path1)
                results2 = dict(pts_filename=file_path2)

                s_points1 = self.points_loader1(results1)['points']
                s_points1.translate(info1['box3d_lidar'][:3])
                s_points_list1.append(s_points1)

                if info1['static_ids']==1:
                    s_points2 = self.points_loader2(results2)['points']
                    s_points2.translate(info1['box3d_lidar'][:3])
                    s_points_list2.append(s_points2)

                group_ids.append(info1['group_id'])
                static_ids.append(info1['static_ids'])

                count += 1

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled1],
                                 dtype=np.long)
            if len(s_points_list2) > 0:
                ret = {
                    'gt_labels_3d':
                    gt_labels,
                    'gt_bboxes_3d':
                    sampled_gt_bboxes,
                    'points':
                    s_points_list1[0].cat(s_points_list1),
                    'points_dense':
                    s_points_list2[0].cat(s_points_list2),
                    'points_fg':
                    s_points_list2[0].cat(s_points_list2),
                    'group_ids':
                    group_ids,
                    'static_ids':
                    np.array(static_ids),
                }
            else:
                ret = {
                    'gt_labels_3d':
                    gt_labels,
                    'gt_bboxes_3d':
                    sampled_gt_bboxes,
                    'points':
                    s_points_list1[0].cat(s_points_list1),
                    'points_dense':
                    s_points_list2,
                    'points_fg':
                    s_points_list2,
                    'group_ids':
                    group_ids,
                    'static_ids':
                    np.array(static_ids),
                }

        return ret

    def sample_class_v2(self, name, num, gt_bboxes):
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        sampled1 = self.sampler_dict1[name].sample(num)
        sampled2 = self.sampler_dict2[name].sample(num)
        sampled1 = copy.deepcopy(sampled1)
        sampled2 = copy.deepcopy(sampled2)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled1)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6])

        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled1], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes[:,:gt_bboxes.shape[1]]], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples1, valid_samples2 = [], []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples1.append(sampled1[i - num_gt])
                valid_samples2.append(sampled2[i - num_gt])
        return valid_samples1, valid_samples2
