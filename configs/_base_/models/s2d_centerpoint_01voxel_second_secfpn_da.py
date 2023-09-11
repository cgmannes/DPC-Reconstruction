voxel_size = [0.1, 0.1, 0.15]
TEACHER_DIR0 = f'../DPC-Reconstruction/work_dirs'
TEACHER_DIR1 = f'/centerpoint_waymo_dense_3class_teacher_detector_seed_00'
TEACHER_DIR2 = f'/2023_09_08_3class_teacher_detector_seed_00'
checkpoint_dense_pth = f'{TEACHER_DIR0}{TEACHER_DIR1}{TEACHER_DIR2}/epoch_30.pth'
model = dict(
    type='S2DCenterPointBaseline',
    checkpoint_dense=checkpoint_dense_pth,
    loss_sdet=dict(
        loss_s2d=dict(
            type='Sparse2DenseMSELoss',
            beta_a=10,
            gamma_a=20,
            beta_b=5,
            gamma_b=20,
        ),
        loss_pcr=None,
        # loss_pcr=dict(
        #     type='Sparse2DensePCRLoss',
        # ),
        loss_distill=dict(
            type='Sparse2DenseDisLoss',
        ),
    ),
    pts_voxel_layer=dict(
        max_num_points=20,
        voxel_size=voxel_size,
        max_voxels=(200000, 200000),
        point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
        # Speed up for dense aggregated point clouds
        deterministic=False,
    ),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1504, 1504],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        training=True,
        pcr=False,
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='S2DCenterHeadBaseline',
        in_channels=sum([256, 256]),
        tasks=[
            dict(num_class=1, class_names=['Car']),
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=[-75.2, -75.2],
            post_center_range=[-84.9, -84.9, -10, 84.9, 84.9, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        # nuScenes
        # loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        # Waymo and CADC
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=2.00),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
            grid_size=[1504, 1504, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=[-75.2, -75.2],
            post_center_limit_range=[-85.2, -85.2, -10, 85.2, 85.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            # # nuScenes eval
            # pre_max_size=1000,
            # post_max_size=83,
            # nms_thr=0.2,
            # # Waymo eval
            pre_max_size=4096,
            post_max_size=500,
            nms_thr=0.25)))
