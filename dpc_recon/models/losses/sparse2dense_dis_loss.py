# Copyright (c) stevewongv/Sparse2Dense. All rights reserved.
# https://github.com/stevewongv/Sparse2Dense
import torch
from mmgen.models.builder import MODULES


@MODULES.register_module()
class Sparse2DenseDisLoss(object):
    """Fast Focal loss for the CenterPoint heatmaps and bboxes
    based on the Sparse2Dense: Learning to Densify 3D Features.
    https://arxiv.org/pdf/2211.13067.pdf
    """
    def __init__(self):
        super().__init__()

    def forward(self, S_preds_dict, T_preds_dict, ind, mask, cat):

        loss_hm_distill = self.fastfocalloss(
            S_preds_dict[0]['heatmap'], T_preds_dict[0]['heatmap'],
            ind, mask, cat)

        return loss_hm_distill

    def _gather_feat(self, feat, ind, mask=None):
        dim  = feat.size(2)
        ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def fastfocalloss(self, out, target, ind, mask, cat):
        '''
        Arguments:
        out, target: B x C x H x W
        ind, mask: B x M
        cat (category id for peaks): B x M
        '''
        mask = mask.sum(dim=-1) / mask.shape[-1]
        mask = mask.float()
        gt = torch.pow(1 - target, 4)
        neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
        neg_loss = neg_loss.sum()

        pos_pred_pix = self._transpose_and_gather_feat(out, ind) # B x M x C
        pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
        num_pos = mask.sum()
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
                mask.unsqueeze(2)
        pos_loss = pos_loss.sum()
        if num_pos == 0:
            return - neg_loss
        return - (pos_loss + neg_loss) / num_pos