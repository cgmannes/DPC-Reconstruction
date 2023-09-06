# Copyright (c) OpenMMLab. All rights reserved.
from .centerpoint import CenterPoint
from .goat_centerpoint import GOATCenterPoint
from .s2d_centerpoint import S2DCenterPoint
from .s2d_centerpoint_baseline import S2DCenterPointBaseline
from .sparse2dense_centerpoint import Sparse2DenseCenterPoint
from .sparse2dense_centerpoint import StudentSparse2DenseCenterPoint
from .sparse2dense_centerpoint import TeacherSparse2DenseCenterPoint

__all__ = [
    'CenterPoint', 'GOATCenterPoint', 'S2DCenterPoint',
    'S2DCenterPointBaseline', 'Sparse2DenseCenterPoint',
    'TeacherSparse2DenseCenterPoint', 'StudentSparse2DenseCenterPoint'
]
