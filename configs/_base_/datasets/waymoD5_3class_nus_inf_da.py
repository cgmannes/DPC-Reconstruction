# dataset settings
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://waymo_data/'))

class_names = [
    'car',
]
point_cloud_range = [-75.2, -75.2, -2, 75.2, 75.2, 4]
input_modality = dict(use_lidar=True, use_camera=False)

sweeps_num = 0
train_pipeline = [
]
test_pipeline = [
    dict(
        type='LoadPointsFromMultiSweeps',
        coord_type='LIDAR',
        dataset_type='Nuscenes',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        shift_height=False,
        use_color=False,
        file_client_args=file_client_args,
        sweeps_num=sweeps_num,
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=False),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(
        type='RemoveGroundPoints',
        sweeps_num=sweeps_num,
        dataset_type='Nuscenes'),
    dict(
        type='ScaleFeaturesMinMax',
        data_range=(0, 255),
        target_range=(0, 1)),
    dict(
        type='GlobalTrans',
        translation=[0, 0, 1.8]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.0,
                flip_ratio_bev_vertical=0.0),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromMultiSweeps',
        coord_type='LIDAR',
        dataset_type='Nuscenes',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        shift_height=False,
        use_color=False,
        file_client_args=file_client_args,
        sweeps_num=sweeps_num,
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=False),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(
        type='RemoveGroundPoints',
        sweeps_num=sweeps_num,
        dataset_type='Nuscenes'),
    dict(
        type='ScaleFeaturesMinMax',
        data_range=(0, 255),
        target_range=(0, 1)),
    dict(
        type='GlobalTrans',
        translation=[0, 0, 1.8]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d',])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=24,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            with_velocity=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        with_velocity=False,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        with_velocity=False,
        box_type_3d='LiDAR'),
    st=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        with_velocity=False,
        box_type_3d='LiDAR'))

evaluation = dict(interval=31, pipeline=eval_pipeline)
