"""
 @Time    : 2021/8/18 22:59
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : 2023_IJCV_OPNet
 @File    : loss.py
 @Function:
 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

###################################################################
# ########################## edge loss #############################
###################################################################
class EdgeLoss(nn.Module):
    def __init__(self, device):
        super(EdgeLoss, self).__init__()
        self.device = device
        laplace = torch.FloatTensor([[-1, -1, -1, ], [-1, 8, -1], [-1, -1, -1]]).view([1, 1, 3, 3]).cuda(self.device)
        # filter shape in Pytorch: out_channel, in_channel, height, width
        self.laplace = nn.Parameter(data=laplace, requires_grad=False).cuda(self.device)
        self.l1_loss = nn.L1Loss()

    def torchLaplace(self, x):
        edge = torch.abs(F.conv2d(x, self.laplace, padding=1)).cuda(self.device)
        return edge

    def forward(self, y_pred, y_true):
        y_true_edge = self.torchLaplace(y_true)
        y_pred = torch.sigmoid(y_pred)
        y_pred_edge = self.torchLaplace(y_pred)
        edge_loss = self.l1_loss(y_pred_edge, y_true_edge)

        return edge_loss


###################################################################
# ########################## iou loss #############################
###################################################################
class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)

        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)


###################################################################
# #################### structure loss #############################
###################################################################
class structure_loss(torch.nn.Module):
    def __init__(self):
        super(structure_loss, self).__init__()

    def _structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter) / (union - inter)
        return (wbce + wiou).mean()

    def forward(self, pred, mask):
        return self._structure_loss(pred, mask)
