_base_ = [
    '../_base_/datasets/waymoD5_3class_da.py',
    '../_base_/models/centerpoint_01voxel_second_secfpn_da.py',
    '../_base_/schedules/cyclic_30e.py',
    '../_base_/default_runtime.py',
]

seed = '00'
sweeps_num = 0
exp_name = f'3class_detector_seed_{seed}_sweeps_{sweeps_num}'
work_dir = f'./work_dirs/centerpoint_waymo_baseline_{exp_name}'