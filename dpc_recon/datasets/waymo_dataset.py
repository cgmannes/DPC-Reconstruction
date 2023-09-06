import os
import copy
import glob
import tempfile
import numpy as np
import mmcv
from os import path as osp
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet3d.datasets import WaymoDataset as _WaymoDataset
from mmdet3d.core.bbox import CameraInstance3DBoxes


@DATASETS.register_module(force=True)
class WaymoDataset(_WaymoDataset):
    def load_annotations(self, ann_file):
        self.data_infos_mtime = os.stat(ann_file).st_mtime_ns
        return super().load_annotations(ann_file)

    def get_data_info(self, index):
        # Reload annotations when the annotations change
        if os.stat(self.ann_file).st_mtime_ns != self.data_infos_mtime:
            self.data_infos = self.load_annotations(self.ann_file)
        input_dict = super().get_data_info(index)
        input_dict.update(
            dict(
                pose=self.data_infos[index]['pose'],
                sweeps=self.data_infos[index]['sweeps'],
                timestamp=self.data_infos[index]['timestamp'] / 1e6,
        ))
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)

        # convert gt_bboxes_3d to velodyne coordinates
        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            self.box_mode_3d, np.linalg.inv(rect @ Trv2c))
        gt_bboxes = annos['bbox']

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]

        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names)

        if 'score' in annos:
            anns_results['gt_scores_3d'] = annos['score'].astype(np.float32)
        if 'ignore' in annos:
            anns_results['gt_ignores_3d'] = annos['ignore'].astype(bool)
        if 'static_ids' in annos:
            anns_results['static_ids'] = annos['static_ids'].astype(int)

        return anns_results

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None,
                       data_format='da'):
        assert ('waymo' in data_format or 'kitti' in data_format or 'da' in data_format), \
            f'invalid data_format {data_format}'

        if 'waymo' in data_format or 'kitti' in data_format:
            return super().format_results(outputs, pklfile_prefix,
                                          submission_prefix, data_format)

        from ..core.evaluation.waymo_utils.prediction_kitti_to_waymo import \
            KITTI2Waymo  # noqa

        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if (not isinstance(outputs[0], dict)) or 'img_bbox' in outputs[0]:
            raise TypeError('Not supported type for reformat results.')
        elif 'pts_bbox' in outputs[0]:
            result_files = dict()
            for name in outputs[0]:
                results_ = [out[name] for out in outputs]
                pklfile_prefix_ = pklfile_prefix + name
                if submission_prefix is not None:
                    submission_prefix_ = f'{submission_prefix}_{name}'
                else:
                    submission_prefix_ = None
                result_files_ = self.bbox2result_kitti(results_, self.CLASSES,
                                                       pklfile_prefix_,
                                                       submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(outputs, self.CLASSES,
                                                  pklfile_prefix,
                                                  submission_prefix)
        waymo_root = osp.join(
            self.data_root.split('kitti_format')[0], 'waymo_format')
        if self.split == 'training':
            waymo_tfrecords_dir = osp.join(waymo_root, 'validation')
            prefix = '1'
        elif self.split == 'testing':
            waymo_tfrecords_dir = osp.join(waymo_root, 'testing')
            prefix = '2'
        else:
            raise ValueError('Not supported split value.')
        save_tmp_dir = tempfile.TemporaryDirectory()
        waymo_results_save_dir = save_tmp_dir.name
        waymo_results_final_path = f'{pklfile_prefix}.bin'
        if 'pts_bbox' in result_files:
            converter = KITTI2Waymo(result_files['pts_bbox'],
                                    waymo_tfrecords_dir,
                                    waymo_results_save_dir,
                                    waymo_results_final_path, prefix)
        else:
            converter = KITTI2Waymo(result_files, waymo_tfrecords_dir,
                                    waymo_results_save_dir,
                                    waymo_results_final_path, prefix)
        converter.convert()
        save_tmp_dir.cleanup()

        return result_files, tmp_dir

    def format_results_v2(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None,
                       data_format='da'):
        assert ('waymo' in data_format or 'kitti' in data_format or 'da' in data_format), \
            f'invalid data_format {data_format}'

        if 'waymo' in data_format or 'kitti' in data_format:
            return super().format_results(outputs, pklfile_prefix,
                                          submission_prefix, data_format)

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

        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        assert(len(outputs) == len(self.data_infos))

        tfrecords_path = self.data_root.replace('kitti_format', 'waymo_format')
        tfrecords_path += 'validation'
        tfrecords = list(sorted(glob.glob(f'{tfrecords_path}/*.tfrecord')))
        context_names = [x.split('-')[1][:-len('_with_camera_labels.tfrecord')] for x in tfrecords]
        assert(len(tfrecords) == 202)

        print('\nConverting prediction to Waymo format')
        objects = metrics_pb2.Objects()
        for result, info in mmcv.track_iter_progress(
                list(zip(outputs, self.data_infos))):
            if 'pts_bbox' in result:
                result = result['pts_bbox']
            boxes = result['boxes_3d']
            scores = result['scores_3d']
            labels = result['labels_3d']

            scene_idx = int(info['image']['image_idx'] // 1000 - 1000)
            context_name = context_names[scene_idx]
            timestamp = info['timestamp']
            for b, s, l in zip(boxes, scores, labels):
                box = label_pb2.Label.Box()
                box.center_x = b[0]
                box.center_y = b[1]
                box.center_z = b[2] + b[5]/2
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

        with open(f'{pklfile_prefix}.bin', 'wb') as f:
            f.write(objects.SerializeToString())

        return None, tmp_dir

    def evaluate(self,
                 results,
                 metric='waymo',
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        assert ('waymo' in metric or 'kitti' in metric or 'da' in metric), \
            f'invalid metric {metric}'
        if 'da' in metric:
            waymo_root = osp.join(
                self.data_root.split('kitti_format')[0], 'waymo_format')
            if pklfile_prefix is None:
                eval_tmp_dir = tempfile.TemporaryDirectory()
                pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
            else:
                eval_tmp_dir = None
            result_files, tmp_dir = self.format_results_v2(
                results,
                pklfile_prefix,
                submission_prefix,
                data_format='da')
            import subprocess
            ret_bytes = subprocess.check_output(
                'simuda/core/evaluation/waymo_utils/' +
                f'compute_detection_metrics_main {pklfile_prefix}.bin ' +
                f'{waymo_root}/gt_static_vel_0.1_points_5_autolab.bin',
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
        else:
            ap_dict = super().__init__(results, metric, logger, pklfile_prefix,
                                       submission_prefix, show, out_dir, pipeline)

        return ap_dict
