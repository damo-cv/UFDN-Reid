import torch
import torch.nn as nn


class CrossEntropyMaskLabel(nn.Module):
    def __init__(self):
        super(CrossEntropyMaskLabel, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, masks=None):
        if masks is None: masks = torch.ones_like(targets).cuda()
        assert masks.size() == targets.size()
        masks = masks.float()
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()
        loss = (- targets * log_probs)
        loss = loss.sum(1)
        loss_masks = loss * masks
        valid_num = masks[masks != 0].size(0)
        eps = 1e-8
        loss_masks = loss_masks.sum(0) / (eps + valid_num)
        return loss_masks

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()
        # self.ce = CrossEntropyMaskLabel()

    def forward(self, input, target, masks=None):
        logp = self.ce(input, target)
        # logp = self.ce(input, target, masks=masks)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()