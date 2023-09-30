import os
import pickle
import multiprocessing as mp
import numpy as np
import torch
from tqdm.autonotebook import tqdm
from scipy.spatial.transform import Rotation as R
from mmdet3d.core import xywhr2xyxyr
from mmdet3d.core.bbox import box_np_ops, LiDARInstance3DBoxes
from mmdet3d.ops.iou3d import boxes_iou_bev, nms_gpu
from nuscenes import NuScenes
from lstk.utils.transforms import affine_transform

delta = np.array([
    # QST val
    # [-0.10990626, -0.00584448,  0.01562548]
    # QST train
    [-0.1196947,  -0.04357472,  0.00318215]
])

if __name__ == '__main__':
    source = 'waymo'
    target = 'nus'
    split = 'train'
    suffix = 'qst'

    limit = 150*40          # 150 objects per frame * 40 frames per scene
    min_samples = 2
    iou_thresh = 0.5

    with open(f'data/nuscenes/nuscenes_infos_{split}.pkl', 'rb') as f:
        infos = pickle.load(f)['infos']
        infos = list(sorted(infos, key=lambda e: e['timestamp']))
    with open(f'work_dirs/centerpoint_{source}_{suffix}/preds_{target}_{split}.pkl', 'rb') as f:
        preds = pickle.load(f)
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes')

    assert(len(infos) == len(preds))
    scene_preds = dict()
    poses = dict()
    poses_inv = dict()
    for info, pts_bbox in zip(infos, preds):
        frame_idx = info['token']
        scene_idx = nusc.get('sample', frame_idx)['scene_token']
        if scene_idx not in scene_preds:
            scene_preds[scene_idx] = []
        if scene_idx not in poses:
            poses[scene_idx] = dict()
            poses_inv[scene_idx] = dict()
        pose = affine_transform(
            rotation=np.roll(info['ego2global_rotation'], -1),
            rotation_format='quat',
            translation=info['ego2global_translation']
        ) @ affine_transform(
            rotation=np.roll(info['lidar2ego_rotation'], -1),
            rotation_format='quat',
            translation=info['lidar2ego_translation']
        )
        poses[scene_idx][frame_idx] = pose
        poses_inv[scene_idx][frame_idx] = np.linalg.inv(pose)

        pts_bbox = pts_bbox['pts_bbox']
        boxes = np.concatenate([
            pts_bbox['boxes_3d'].tensor.numpy(),
            pts_bbox['labels_3d'][:,None],
            pts_bbox['scores_3d'][:,None],
            np.full((len(pts_bbox['boxes_3d']),1), -1)], axis=1)
        boxes[:,:3] = boxes[:,:3] @ pose[:3,:3].T
        boxes[:,:3] += pose[:3,3]
        boxes[:,6] -= R.from_matrix(pose[:3,:3]).as_euler('zyx')[0]
        scene_preds[scene_idx].append(boxes)
    for scene_idx in scene_preds:
        scene_preds[scene_idx] = np.concatenate(scene_preds[scene_idx])
    with open(f'work_dirs/centerpoint_{source}_{suffix}/scene_preds_{target}_{split}.pkl', 'wb') as f:
        pickle.dump(scene_preds, f)

    obj_id = 0
    scene_preds_filtered = dict()
    for scene_id, boxes in tqdm(scene_preds.items(), desc='Cluster + filter'):
        scene_preds_filtered[scene_id] = []
        boxes = np.array(boxes)
        # Sort by scores
        boxes = torch.tensor(boxes[np.argsort(boxes[:,8])[::-1]][:limit]).float().cuda()
        boxes2d = xywhr2xyxyr(boxes[:,[0,1,3,4,6]])
        iou_valid = boxes_iou_bev(boxes2d, boxes2d)
        iou_valid.fill_diagonal_(1.0)
        iou_valid = iou_valid > iou_thresh
        boxes_valid = (iou_valid.sum(dim=1) >= min_samples)
        boxes = boxes[boxes_valid]
        iou_valid = iou_valid[boxes_valid][:,boxes_valid]
        while boxes.shape[0] > 0:
            iou_valid_i = iou_valid[0]
            iou_invalid_i = ~iou_valid_i
            box_i = boxes[iou_valid_i]
            if box_i.shape[0] >= min_samples:
                # Select most confident heading
                heading_i = box_i[box_i[:,8].argmax(),6]
                # Select most common label
                label_i = box_i[:,7].mode()[0]
                # Average scores
                score_i = box_i[:,8].mean()
                # Weighted average box location and size
                box_i = (box_i*(box_i[:,8]/box_i[:,8].sum()).unsqueeze(-1)).sum(0)
                # Overwrite box data
                box_i[6] = heading_i
                box_i[7] = label_i
                box_i[8] = score_i
                # Unique tracking id for objects, -1 indicates prediction
                # comes from sparse model
                box_i[9] = obj_id
                obj_id += 1
                scene_preds_filtered[scene_id].append(box_i)
            boxes = boxes[iou_invalid_i.squeeze()]
            iou_valid = iou_valid[iou_invalid_i][:,iou_invalid_i]
        if len(scene_preds_filtered[scene_id]) > 0:
            boxes = torch.stack(scene_preds_filtered[scene_id])
            boxes2d = xywhr2xyxyr(boxes[:,[0,1,3,4,6]])
            keep = nms_gpu(boxes2d, boxes[:,8], 0.1)
            scene_preds_filtered[scene_id] = boxes[keep].cpu().numpy()
        else:
            scene_preds_filtered[scene_id] = np.zeros((0, 10), dtype=np.float32)
        torch.cuda.empty_cache()
    with open(f'work_dirs/centerpoint_{source}_{suffix}/scene_preds_{target}_{split}_filtered.pkl', 'wb') as f:
        pickle.dump(scene_preds_filtered, f)

    boxes_sn = []
    with open(f'work_dirs/centerpoint_{source}_{target}_sn/preds_{target}_{split}.pkl', 'rb') as f:
        for pts_bbox in pickle.load(f):
            pts_bbox = pts_bbox['pts_bbox']
            boxes_sn.append(np.concatenate([
                pts_bbox['boxes_3d'].tensor.numpy(),
                pts_bbox['labels_3d'][:,None],
                pts_bbox['scores_3d'][:,None],
                np.full((len(pts_bbox['boxes_3d']),1), -1)], axis=1))

    def get_frame_boxes(args):
        info, boxes_sparse_i = args
        frame_idx = info['token']
        scene_idx = nusc.get('sample', frame_idx)['scene_token']
        pose = poses_inv[scene_idx][frame_idx]

        boxes = scene_preds_filtered[scene_idx].copy()
        boxes[:,:3] = boxes[:,:3] @ pose[:3,:3].T
        boxes[:,:3] += pose[:3,3]
        boxes[:,6] -= R.from_matrix(pose[:3,:3]).as_euler('zyx')[0]
        boxes[:,6] = boxes[:,6] % (2*np.pi)
        boxes[boxes[:,6] > np.pi,6] -= 2*np.pi
        boxes = boxes[np.linalg.norm(boxes[:,:2], axis=1) < 80]
        for c in range(1):
            boxes[boxes[:,7]==c,3:6] += delta[c]

        boxes = np.concatenate([boxes, boxes_sparse_i])
        # Remove predictions without points
        boxes2 = boxes.copy()
        boxes2[:,2] -= 1.8
        points = np.fromfile(info['lidar_path'], dtype=np.float32).reshape(-1, 5)
        num_pts = box_np_ops.points_in_rbbox(points, boxes2, origin=(0.5, 0.5, 0.0)).sum(0)
        boxes = boxes[num_pts > 1]
        return boxes

    preds_filtered = []
    with mp.Pool(os.cpu_count()) as p:
        imap = p.imap(get_frame_boxes, zip(infos, boxes_sn), chunksize=32)
        for boxes in tqdm(imap, total=len(infos), desc='Generate final predictions'):
            # NMS
            boxes_cuda = torch.tensor(boxes).float().cuda()
            boxes2d = xywhr2xyxyr(boxes_cuda[:,[0,1,3,4,6]])
            keep = nms_gpu(boxes2d, boxes_cuda[:,8], 0.1)
            boxes = boxes[keep.cpu().numpy()]
            boxes = boxes[np.argsort(boxes[:,8])[::-1]][:200]

            pts_bbox = dict()
            pts_bbox['boxes_3d'] = LiDARInstance3DBoxes(torch.tensor(boxes[:,:7]).float())
            pts_bbox['labels_3d'] = torch.tensor(boxes[:,7]).int()
            pts_bbox['scores_3d'] = torch.tensor(boxes[:,8]).float()
            pts_bbox['obj_ids'] = torch.tensor(boxes[:,9]).float()
            preds_filtered.append(dict(pts_bbox=pts_bbox))

    # Without multi-processing
    # for args in tqdm(list(zip(infos, boxes_sparse))):
    #     preds_filtered.append(get_frame_boxes(args))

    with open(f'work_dirs/centerpoint_{source}_{suffix}/preds_{target}_{split}_scp.pkl', 'wb') as f:
        pickle.dump(preds_filtered, f)
