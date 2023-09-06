# Copyright (c) stevewongv/Sparse2Dense. All rights reserved.
# https://github.com/stevewongv/Sparse2Dense
import torch
import torch.nn.functional as F

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def fastfocalloss(out, target, ind, mask, cat, ignore_mask=None):
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
    if ignore_mask is None:
        neg_loss = neg_loss.sum()
    else:
        ignore_inds = ignore_mask > 0
        neg_loss = neg_loss[ignore_inds].sum()

    pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos

def mse_loss(f_1, f_2):
    feat_1 = f_1.dense()
    N, C, D, H, W = feat_1.shape
    feat_1 = feat_1.view(N, C*D, H, W)
    feat_2 = f_2.dense().view(N, C*D, H, W)
    return F.mse_loss(feat_1, feat_2.detach(), reduction='sum') / (f_2.features.shape[0] * 10)

def distill_reg_loss(output, target, mask, ind):
    pred = _transpose_and_gather_feat(output, ind)
    gt = _transpose_and_gather_feat(target, ind)
    mask = mask.float()#.unsqueeze(2)

    loss = F.smooth_l1_loss(pred*mask,  gt*mask, reduction='none')
    mask = mask.sum(dim=-1) / mask.shape[-1]
    loss = loss / (mask.sum() + 1e-4)
    loss = loss.transpose(2, 0).sum(dim=2).sum(dim=1)
    return loss