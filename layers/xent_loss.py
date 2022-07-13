import torch
from torch import nn
import numpy as np

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# class CrossEntropyLabelSmooth(nn.Module):
#     """Cross entropy loss with label smoothing regularizer.

#     Reference:
#     Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
#     Equation: y = (1 - epsilon) * y + epsilon / K.

#     Args:
#         num_classes (int): number of classes.
#         epsilon (float): weight.
#     """
#     def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
#         super(CrossEntropyLabelSmooth, self).__init__()
#         self.num_classes = num_classes
#         self.epsilon = epsilon
#         self.use_gpu = use_gpu
#         self.logsoftmax = nn.LogSoftmax(dim=1)

#     def forward(self, inputs, targets, masks=None):
#         """
#         Args:
#             inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
#             targets: ground truth labels with shape (num_classes)
#         """
#         if masks is None: masks = torch.ones_like(targets).float().cuda()
#         log_probs = self.logsoftmax(inputs)
#         targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()
#         targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
#         loss = - targets * log_probs
#         loss = loss.sum(1)
#         loss_masks = loss * masks
#         valid_num = masks[masks != 0].size(0) 
#         eps = 1e-8
#         loss_masks = loss_masks.sum() / (eps + valid_num)
#         return loss_masks

# class CrossEntropySoftLabel(nn.Module):
#     """Cross entropy loss with label smoothing regularizer.

#     Reference:
#     Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
#     Equation: y = (1 - epsilon) * y + epsilon / K.

#     Args:
#         num_classes (int): number of classes.
#         epsilon (float): weight.
#     """
#     def __init__(self):
#         super(CrossEntropySoftLabel, self).__init__()
#         self.logsoftmax = nn.LogSoftmax(dim=1)

#     def forward(self, inputs, target):
#         """
#         Args:
#             inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
#             targets: ground truth labels with shape (num_classes)
#         """
#         # if masks is None: masks = torch.ones(targets.size(0)).cuda()
#         log_probs = self.logsoftmax(inputs)
#         loss = - targets * log_probs
#         loss = loss.sum(1)
#         # loss_masks = loss * masks
#         # valid_num = masks[masks != 0].size(0) 
#         # eps = 1e-8
#         # loss_masks = loss_masks.sum() / 
#         return loss