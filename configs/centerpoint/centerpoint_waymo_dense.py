_base_ = [
    '../_base_/datasets/waymoD5_3class_dense_da.py',
    '../_base_/models/centerpoint_01voxel_second_secfpn_dense_da.py',
    '../_base_/schedules/cyclic_30e.py',
    '../_base_/default_runtime.py',
]

seed = '00'
exp_name = f'3class_teacher_detector_seed_{seed}'
work_dir = f'./work_dirs/centerpoint_waymo_dense_{exp_name}'