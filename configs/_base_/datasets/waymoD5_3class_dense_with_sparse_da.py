# dataset settings
# D5 in the config name means the whole dataset is divided into 5 folds
# We only use one fold for efficient experiments
dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format/'
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://waymo_data/'))

class_names = [
    'Car',
]
point_cloud_range = [-75.2, -75.2, -2, 75.2, 75.2, 4]
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'waymo_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            Car=5)),
    classes=class_names,
    sample_groups=dict(
        Car=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

sweeps_num = 0
train_pipeline = [
    dict(
        type='LoadPointsFromMultiSweeps',
        coord_type='LIDAR',
        dataset_type='Waymo',
        load_dim=6,
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
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RemoveGroundPoints',
        sweeps_num=sweeps_num,
        dataset_type='Waymo'),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    # Create a copy of sparse point cloud to prepare loading for dense
    dict(type='CopyData', src='points', dst='points_dense'),
    dict(
        type='RemoveObjectPoints',
        sweeps_num=sweeps_num,
        points_name='points_dense',
        dataset_type='Waymo',
        coord_type='LIDAR'),
    dict(
        type='LoadAggregatedPoints',
        data_root='data/lstk/complete/waymo',
        metadata_path='data/lstk/complete/waymo/metadata_train.pkl',
        dbinfos_path='data/waymo/kitti_format/waymo_dbinfos_train_autolab_3class.pkl',
        load_scene=False,
        load_dims=[3, 3],
        use_dims=[0, 1, 2, 3, 4],
        load_as=['points_dense', 'points_fg']),
    dict(
        type='ScaleFeaturesTanh',
        points_name=['points', 'points_dense', 'points_fg']),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='PointShuffle', points_name='points_dense'),
    dict(type='PointShuffle', points_name='points_fg'),
    dict(type='DefaultFormatBundle3D', class_names=class_names,
         format_cfg=dict(points_dense={}, points_fg={})),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d',
                                 'points_dense', 'points_fg'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromMultiSweeps',
        coord_type='LIDAR',
        dataset_type='Waymo',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4],
        shift_height=False,
        use_color=False,
        file_client_args=file_client_args,
        sweeps_num=sweeps_num,
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=False),
    dict(
        type='RemoveGroundPoints',
        sweeps_num=sweeps_num,
        dataset_type='Waymo'),
    dict(type='ScaleFeaturesTanh'),
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
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromMultiSweeps',
        coord_type='LIDAR',
        dataset_type='Waymo',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4],
        shift_height=False,
        use_color=False,
        file_client_args=file_client_args,
        sweeps_num=sweeps_num,
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=False),
    dict(
        type='RemoveGroundPoints',
        sweeps_num=sweeps_num,
        dataset_type='Waymo'),
    dict(type='ScaleFeaturesTanh'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=24,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'waymo_infos_train.pkl',
            split='training',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            # load one frame every five frames
            load_interval=10)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'waymo_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'waymo_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

evaluation = dict(interval=31, pipeline=eval_pipeline)
