# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models import DETECTORS
from mmdet3d.models.detectors import CenterPoint


@DETECTORS.register_module()
class Sparse2DenseCenterPoint(CenterPoint):
    """
    A helper class that extends MMDet3d CenterPoint to implement the functionality
    of CenterPoint but return additional features for S2D losses
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points"""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        y = self.pts_backbone(x)
        if self.with_pts_neck:
            y = self.pts_neck(y)
        return y, x

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points"""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats, pts_middle_enc_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats, pts_middle_enc_feats)


@DETECTORS.register_module()
class TeacherSparse2DenseCenterPoint(Sparse2DenseCenterPoint):
    """
    A helper class that extends MMDet3d CenterPoint to implement the functionality
    of CenterPoint but return additional features for S2D losses
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def extract_pts_feat(self, pts, img_feats, img_metas, main_pipeline):
        """Extract features of points"""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x, input_sp_tensor = self.pts_middle_encoder(voxel_features, coors, batch_size)
        if main_pipeline:
            voxel_ret = None
            y = self.pts_backbone(x)
            if self.with_pts_neck:
                y = self.pts_neck(y)
        else:
            voxel_ret = [voxel_features, coors, batch_size, input_sp_tensor.dense()]
            y = None
        return y, x, voxel_ret

    def extract_feat(self, points, img, img_metas, main_pipeline=True):
        """Extract features from images and points"""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats, pts_middle_enc_feats, voxel_ret = self.extract_pts_feat(
            points, img_feats, img_metas, main_pipeline)
        return img_feats, pts_feats, pts_middle_enc_feats, voxel_ret


@DETECTORS.register_module()
class StudentSparse2DenseCenterPoint(Sparse2DenseCenterPoint):
    """
    A helper class that extends MMDet3d CenterPoint to implement the functionality
    of CenterPoint but return additional features for S2D losses
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points"""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x, _ = self.pts_middle_encoder(voxel_features, coors, batch_size)
        N, _, H, W = x.shape
        F_a, F_b = self.pts_neck.forward_s2d(x)
        y = self.pts_backbone(F_a)
        if self.with_pts_neck:
            y = self.pts_neck(y)
            pcr_ret = self.pts_neck.forward_pcr(F_b, N=N, _=_, H=H, W=W)
        return y, F_a, F_b, pcr_ret

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points"""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats, F_a, F_b, pcr_ret = self.extract_pts_feat(points, img_feats, img_metas)
        return img_feats, pts_feats, F_a, F_b, pcr_ret