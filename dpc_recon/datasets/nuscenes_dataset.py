import tempfile
import numpy as np
import torch
import mmcv
from os import path as osp
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import NuScenesDataset as _NuScenesDataset
from lstk.utils.transforms import affine_transform


@DATASETS.register_module(force=True)
class NuScenesDataset(_NuScenesDataset):
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        lidar2ego_rotation = np.array(info['lidar2ego_rotation'])
        lidar2ego_translation = np.array(info['lidar2ego_translation'])
        ego2global_rotation = np.array(info['ego2global_rotation'])
        ego2global_translation = np.array(info['ego2global_translation'])

        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            lidar2ego=affine_transform(
                rotation=np.roll(lidar2ego_rotation, -1),
                rotation_format='quat',
                translation=lidar2ego_translation,
            ),
            ego2global=affine_transform(
                rotation=np.roll(ego2global_rotation, -1),
                rotation_format='quat',
                translation=ego2global_translation,
            ),
        )

        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        ann_info = super().get_ann_info(index)
        ann_info['sweeps'] = []
        for sweep_info in info['sweeps']:
            # if 'gt_boxes' not in sweep_info or 'gt_names' not in sweep_info:
            #     continue
            # gt_bboxes_3d = sweep_info['gt_boxes']
            # gt_names_3d = sweep_info['gt_names']
            # gt_labels_3d = []
            # for cat in gt_names_3d:
            #     if cat in self.CLASSES:
            #         gt_labels_3d.append(self.CLASSES.index(cat))
            #     else:
            #         gt_labels_3d.append(-1)
            # gt_labels_3d = np.array(gt_labels_3d)
            timestamp = sweep_info['timestamp']
            sensor2ego_rotation = np.array(sweep_info['sensor2ego_rotation'])
            sensor2ego_translation = np.array(sweep_info['sensor2ego_translation'])
            ego2global_rotation = np.array(sweep_info['ego2global_rotation'])
            ego2global_translation = np.array(sweep_info['ego2global_translation'])

            # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
            # the same as KITTI (0.5, 0.5, 0)
            # gt_bboxes_3d = LiDARInstance3DBoxes(
            #     gt_bboxes_3d,
            #     box_dim=gt_bboxes_3d.shape[-1],
            #     origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

            ann_info['sweeps'].append(
                dict(
                    # gt_bboxes_3d=gt_bboxes_3d,
                    # gt_labels_3d=gt_labels_3d,
                    # gt_names=gt_names_3d,
                    timestamp = timestamp,
                    sensor2ego=affine_transform(
                        rotation=np.roll(sensor2ego_rotation, -1),
                        rotation_format='quat',
                        translation=sensor2ego_translation,
                    ),
                    ego2global=affine_transform(
                        rotation=np.roll(ego2global_rotation, -1),
                        rotation_format='quat',
                        translation=ego2global_translation,
                    ),
            ))
        return ann_info

    def format_results(self,
                       results,
                       jsonfile_prefix=None,
                       data_format='nuscenes'):
        assert ('nuscenes' in data_format or 'waymo' in data_format), \
            f'invalid data_format {data_format}'

        if 'nuscenes' in data_format:
            return super().format_results(results, jsonfile_prefix)

        if 'waymo' in data_format:
            from nuscenes import NuScenes
            from waymo_open_dataset import label_pb2
            from waymo_open_dataset.protos import metrics_pb2
            k2w_cls_map = {
                'car': label_pb2.Label.TYPE_VEHICLE,
                'pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
                'sign': label_pb2.Label.TYPE_SIGN,
                'bicycle': label_pb2.Label.TYPE_CYCLIST,
                'truck': label_pb2.Label.TYPE_TRUCK,
                'bus': label_pb2.Label.TYPE_BUS,
                'motorcycle': label_pb2.Label.TYPE_MOTORCYCLE
            }

            if jsonfile_prefix is None:
                tmp_dir = tempfile.TemporaryDirectory()
                jsonfile_prefix = osp.join(tmp_dir.name, 'results')
            else:
                tmp_dir = None

            assert(len(results) == len(self.data_infos))
            version = 'v1.0-trainval'
            nusc = NuScenes(version=version, dataroot=self.data_root)

            print('\nConverting prediction to Waymo format')
            objects = metrics_pb2.Objects()
            for result, info in mmcv.track_iter_progress(
                    list(zip(results, self.data_infos))):
                if 'pts_bbox' in result:
                    result = result['pts_bbox']
                boxes = result['boxes_3d']
                scores = result['scores_3d']
                labels = result['labels_3d']

                sample = nusc.get('sample', info['token'])
                context_name = sample['scene_token']
                timestamp = sample['timestamp']
                for b, s, l in zip(boxes, scores, labels):
                    box = label_pb2.Label.Box()
                    box.center_x = b[0]
                    box.center_y = b[1]
                    box.center_z = b[2] + b[5]/2 - 1.8
                    box.width = b[3]
                    box.length = b[4]
                    box.height = b[5]
                    box.heading = -b[6] - np.pi/2

                    o = metrics_pb2.Object()
                    o.object.box.CopyFrom(box)
                    o.object.type = k2w_cls_map[self.CLASSES[l]]
                    o.score = s

                    o.context_name = context_name
                    o.frame_timestamp_micros = timestamp

                    objects.objects.append(o)

            with open(f'{jsonfile_prefix}.bin', 'wb') as f:
                f.write(objects.SerializeToString())

            return tmp_dir

    def evaluate(self,
                 results,
                 metric='nuscenes',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        assert ('nuscenes' in metric or 'waymo' in metric), \
            f'invalid metric {metric}'

        if 'nuscenes' in metric:
            if not self.with_velocity:
                for result in results:
                    boxes3d = result['pts_bbox']['boxes_3d'].tensor
                    boxes3d[:,2] -= 1.8
                    velos = torch.zeros((boxes3d.shape[0], 2),
                                        dtype=boxes3d.dtype, device=boxes3d.device)
                    result['pts_bbox']['boxes_3d'].tensor = torch.cat(
                        [boxes3d, velos], dim=1)
            return super().evaluate(results, metric, logger, jsonfile_prefix,
                                    result_names, show, out_dir, pipeline)

        if 'waymo' in metric:
            if jsonfile_prefix is None:
                eval_tmp_dir = tempfile.TemporaryDirectory()
                jsonfile_prefix = osp.join(eval_tmp_dir.name, 'results')
            else:
                eval_tmp_dir = None
            tmp_dir = self.format_results(results, jsonfile_prefix, 'waymo')

            import subprocess
            ret_bytes = subprocess.check_output(
                'simuda/core/evaluation/waymo_utils/' +
                f'compute_detection_metrics_main {jsonfile_prefix}.bin ' +
                f'{self.data_root}/gt.bin',
                shell=True)
            ret_texts = ret_bytes.decode('utf-8')
            print_log(ret_texts)
            # parse the text to get ap_dict
            ap_dict = {
                'Vehicle/L1 mAP': 0,
                'Vehicle/L1 mAPH': 0,
                'Vehicle/L2 mAP': 0,
                'Vehicle/L2 mAPH': 0,
                'Pedestrian/L1 mAP': 0,
                'Pedestrian/L1 mAPH': 0,
                'Pedestrian/L2 mAP': 0,
                'Pedestrian/L2 mAPH': 0,
                'Sign/L1 mAP': 0,
                'Sign/L1 mAPH': 0,
                'Sign/L2 mAP': 0,
                'Sign/L2 mAPH': 0,
                'Cyclist/L1 mAP': 0,
                'Cyclist/L1 mAPH': 0,
                'Cyclist/L2 mAP': 0,
                'Cyclist/L2 mAPH': 0,
                'Truck/L1 mAP': 0,
                'Truck/L1 mAPH': 0,
                'Truck/L2 mAP': 0,
                'Truck/L2 mAPH': 0,
                'Bus/L1 mAP': 0,
                'Bus/L1 mAPH': 0,
                'Bus/L2 mAP': 0,
                'Bus/L2 mAPH': 0,
                'Motorcycle/L1 mAP': 0,
                'Motorcycle/L1 mAPH': 0,
                'Motorcycle/L2 mAP': 0,
                'Motorcycle/L2 mAPH': 0
            }
            mAP_splits = ret_texts.split('mAP ')
            mAPH_splits = ret_texts.split('mAPH ')
            for idx, key in enumerate(ap_dict.keys()):
                split_idx = int(idx / 2) + 1
                if idx % 2 == 0:  # mAP
                    ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
                else:  # mAPH
                    ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])
            if eval_tmp_dir is not None:
                eval_tmp_dir.cleanup()

            if tmp_dir is not None:
                tmp_dir.cleanup()

            if show:
                raise NotImplementedError
            return ap_dict
