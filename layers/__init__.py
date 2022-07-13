# encoding: utf-8


import torch.nn.functional as F
import torch.nn as nn

from .triplet_loss import TripletLoss
from .focal_loss import FocalLoss
from .xent_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .ns_structure_loss import NSSD_loss,NSSD_loss_2,NSSD_loss_3

def l2_loss(x, y):
    N = x.size(0)
    z = (x-y) * (x-y)
    return torch.sum(z) / N

def l1_loss(x, y):
    N = x.size(0)
    z = torch.abs(x-y)
    return torch.sum(z) / N

def make_loss(cfg, num_classes): 
    focal_loss = FocalLoss(gamma=2)
    triplet_loss = TripletLoss(cfg.SOLVER.MARGIN) 
    xent_loss = CrossEntropyLabelSmooth(num_classes=num_classes)   

    def loss_func(score, feat, targets, tids):
        if feat.size(1)==2048:
            loss_dict=cfg.MODEL.LOSS
        else:
            loss_dict=cfg.MODEL.LOSS_LOCAL
        loss=0
        for i, l in enumerate(loss_dict):
            if cfg.MODEL.LOSS_WEIGHT[i] == 0: continue
            if l == 'focal': loss += cfg.MODEL.LOSS_WEIGHT[i] * focal_loss(score, targets)
            if l == 'triplet': loss += cfg.MODEL.LOSS_WEIGHT[i] * triplet_loss(feat, targets)[0]
            if l == 'xent': loss += cfg.MODEL.LOSS_WEIGHT[i] * xent_loss(score, targets)
            if l == 'softmax': loss += cfg.MODEL.LOSS_WEIGHT[i] * F.cross_entropy(score, targets)
        return loss
    return loss_func