import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from torchvision import transforms
from functools import partial
from operator import itemgetter
from mmseg.registry import MODELS


class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float()  # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


# Tools
def kl_div(a, b):  # q,p
    return F.softmax(b, dim=1) * (F.log_softmax(b, dim=1) - F.log_softmax(a, dim=1))


def one_hot2dist(seg):
    res = np.zeros_like(seg)
    for i in range(len(seg)):
        posmask = seg[i].astype(np.bool_)
        if posmask.any():
            negmask = ~posmask
            res[i] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


def class2one_hot(seg, C):
    seg = seg.unsqueeze(dim=0) if len(seg.shape) == 2 else seg
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    return res


@MODELS.register_module()
class ABL(nn.Module):
    def __init__(self, isdetach=True, max_N_ratio=1 / 100, ignore_label=255, label_smoothing=0.2, weight=None,
                 max_clip_dist=20.):
        super(ABL, self).__init__()
        self.ignore_label = ignore_label
        self.label_smoothing = label_smoothing
        self.isdetach = isdetach
        self.max_N_ratio = max_N_ratio

        self.weight_func = lambda w, max_distance=max_clip_dist: torch.clamp(w, max=max_distance) / max_distance

        self.dist_map_transform = transforms.Compose([
            lambda img: img.unsqueeze(0),
            lambda nd: nd.type(torch.int64),
            partial(class2one_hot, C=1),
            itemgetter(0),
            lambda t: t.cpu().numpy(),
            one_hot2dist,
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

        if label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=ignore_label,
                reduction='none'
            )
        else:
            self.criterion = LabelSmoothSoftmaxCEV1(
                reduction='none',
                ignore_index=ignore_label,
                lb_smooth=label_smoothing
            )

        self.loss_name_ = 'loss_abl'

    def logits2boundary(self, logit):
        eps = 1e-5
        _, _, h, w = logit.shape
        max_N = (h * w) * self.max_N_ratio
        kl_ud = kl_div(logit[:, :, 1:, :], logit[:, :, :-1, :]).sum(1, keepdim=True)
        kl_lr = kl_div(logit[:, :, :, 1:], logit[:, :, :, :-1]).sum(1, keepdim=True)
        kl_ud = torch.nn.functional.pad(
            kl_ud, [0, 0, 0, 1, 0, 0, 0, 0], mode='constant', value=0)
        kl_lr = torch.nn.functional.pad(
            kl_lr, [0, 1, 0, 0, 0, 0, 0, 0], mode='constant', value=0)
        kl_combine = kl_lr + kl_ud
        while True:  # avoid the case that full image is the same color
            kl_combine_bin = (kl_combine > eps).to(torch.float)
            if kl_combine_bin.sum() > max_N:
                eps *= 1.2
            else:
                break
        # dilate
        dilate_weight = torch.ones((1, 1, 3, 3)).cuda()
        edge2 = torch.nn.functional.conv2d(kl_combine_bin, dilate_weight, stride=1, padding=1)
        edge2 = edge2.squeeze(1)  # NCHW->NHW
        kl_combine_bin = (edge2 > 0)
        return kl_combine_bin

    def gt2boundary(self, gt, ignore_label=-1):  # gt NHW
        gt_ud = gt[:, 1:, :] - gt[:, :-1, :]  # NHW
        gt_lr = gt[:, :, 1:] - gt[:, :, :-1]
        gt_ud = torch.nn.functional.pad(gt_ud, [0, 0, 0, 1, 0, 0], mode='constant', value=0) != 0
        gt_lr = torch.nn.functional.pad(gt_lr, [0, 1, 0, 0, 0, 0], mode='constant', value=0) != 0
        gt_combine = gt_lr + gt_ud
        del gt_lr
        del gt_ud

        # set 'ignore area' to all boundary
        gt_combine += (gt == ignore_label)

        return gt_combine > 0

    def get_direction_gt_predkl(self, pred_dist_map, pred_bound, logits):
        # NHW,NHW,NCHW
        eps = 1e-5
        # bound = torch.where(pred_bound)  # 3k
        bound = torch.nonzero(pred_bound * 1)
        n, x, y = bound.T
        max_dis = 1e5

        logits = logits.permute(0, 2, 3, 1)  # NHWC

        pred_dist_map_d = torch.nn.functional.pad(pred_dist_map, (1, 1, 1, 1, 0, 0), mode='constant',
                                                  value=max_dis)  # NH+2W+2

        logits_d = torch.nn.functional.pad(logits, (0, 0, 1, 1, 1, 1, 0, 0), mode='constant')  # N(H+2)(W+2)C
        logits_d[:, 0, :, :] = logits_d[:, 1, :, :]  # N(H+2)(W+2)C
        logits_d[:, -1, :, :] = logits_d[:, -2, :, :]  # N(H+2)(W+2)C
        logits_d[:, :, 0, :] = logits_d[:, :, 1, :]  # N(H+2)(W+2)C
        logits_d[:, :, -1, :] = logits_d[:, :, -2, :]  # N(H+2)(W+2)C

        """
        | 4| 0| 5|
        | 2| 8| 3|
        | 6| 1| 7|
        """
        x_range = [1, -1, 0, 0, -1, 1, -1, 1, 0]
        y_range = [0, 0, -1, 1, 1, 1, -1, -1, 0]
        dist_maps = torch.zeros((0, len(x))).cuda()  # 8k
        kl_maps = torch.zeros((0, len(x))).cuda()  # 8k

        kl_center = logits[(n, x, y)]  # KC

        for dx, dy in zip(x_range, y_range):
            dist_now = pred_dist_map_d[(n, x + dx + 1, y + dy + 1)]
            dist_maps = torch.cat((dist_maps, dist_now.unsqueeze(0)), 0)

            if dx != 0 or dy != 0:
                logits_now = logits_d[(n, x + dx + 1, y + dy + 1)]
                # kl_map_now = torch.kl_div((kl_center+eps).log(), logits_now+eps).sum(2)  # 8KC->8K
                if self.isdetach:
                    logits_now = logits_now.detach()
                kl_map_now = kl_div(kl_center, logits_now)

                kl_map_now = kl_map_now.sum(1)  # KC->K
                kl_maps = torch.cat((kl_maps, kl_map_now.unsqueeze(0)), 0)
                torch.clamp(kl_maps, min=0.0, max=20.0)

        # direction_gt shound be Nk  (8k->K)
        direction_gt = torch.argmin(dist_maps, dim=0)
        # weight_ce = pred_dist_map[bound]
        weight_ce = pred_dist_map[(n, x, y)]
        # print(weight_ce)

        # delete if min is 8 (local position)
        direction_gt_idx = [direction_gt != 8]
        direction_gt = direction_gt[direction_gt_idx]

        kl_maps = torch.transpose(kl_maps, 0, 1)
        direction_pred = kl_maps[direction_gt_idx]
        weight_ce = weight_ce[direction_gt_idx]

        return direction_gt, direction_pred, weight_ce

    def get_dist_maps(self, target):
        target_detach = target.clone().detach()
        dist_maps = torch.cat([self.dist_map_transform(target_detach[i]) for i in range(target_detach.shape[0])])
        out = -dist_maps
        out = torch.where(out > 0, out, torch.zeros_like(out))

        return out

    def forward(self, logits, target, weight=None, ignore_index=None):
        eps = 1e-10
        ph, pw = logits.size(2), logits.size(3)
        h, w = target.size(1), target.size(2)

        if ph != h or pw != w:
            logits = F.interpolate(input=logits, size=(
                h, w), mode='bilinear', align_corners=True)

        gt_boundary = self.gt2boundary(target, ignore_label=self.ignore_label)

        dist_maps = self.get_dist_maps(
            gt_boundary).cuda()  # <-- it will slow down the training, you can put it to dataloader.

        pred_boundary = self.logits2boundary(logits)
        if pred_boundary.sum() < 1:  # avoid nan
            return None  # you should check in the outside. if None, skip this loss.

        direction_gt, direction_pred, weight_ce = self.get_direction_gt_predkl(dist_maps, pred_boundary,
                                                                               logits)  # NHW,NHW,NCHW

        # direction_pred [K,8], direction_gt [K]
        loss = self.criterion(direction_pred, direction_gt)  # careful

        weight_ce = self.weight_func(weight_ce)
        loss = (loss * weight_ce).mean()  # add distance weight

        return loss

    @property
    def loss_name(self):
        return self.loss_name_

# if __name__ == '__main__':
#     from torch.backends import cudnn
#     import os
#     import random
#
#     cudnn.benchmark = False
#     cudnn.deterministic = True
#
#     seed = 0
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#     random.seed(seed)
#     np.random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
#     n, c, h, w = 1, 2, 100, 100
#     gt = torch.zeros((n, h, w)).cuda()
#     gt[0, 5] = 1
#     gt[0, 50] = 1
#     logits = torch.randn((n, c, h, w)).cuda()
#
#     abl = ABL()
#     print(abl(logits, gt))
