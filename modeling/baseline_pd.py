#encoding: utf-8


import cv2
import numpy as np

import torch
from torch import nn
import torchvision.utils as vutils
import torchvision.transforms as T
from torch.nn import functional as F
import torch.nn.utils.weight_norm as weightNorm
from torch.nn.parameter import Parameter

import sys
import copy
sys.path.append('..')

from .backbone.resnet import ResNet, BasicBlock, Bottleneck
from .backbone.PDecouple_swin import PSwinTransformer
from .backbone.PDecouple_resnet import Presnet

BN = None

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_reduction(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.constant_(m.weight, 1.0 / m.weight.size(0))
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class SpatialAttn(nn.Module):
    """Spatial Attention Layer"""
    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x):
        # global cross-channel averaging # e.g. 32,2048,24,8
        x = x.mean(1, keepdim=True)  # e.g. 32,1,24,8
        h = x.size(2)
        w = x.size(3)
        x = x.view(x.size(0),-1)     # e.g. 32,192
        z = x
        # for b in range(x.size(0)):
        #     z[b] /= torch.sum(z[b])
        z = 10*self.softmax_layer(z)
        z = z.view(x.size(0), 1, h, w)
        return z

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv1 = nn.Conv2d(2, 1, 1, padding=0, bias=False)
        # self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        # self.conv1 = nn.Conv2d(1, 1, 1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def simple_group_split(world_size, rank, num_groups):
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(link.new_group(rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank//group_size]

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck, bias=False)] 
        add_block += [BN(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
            #add_block += [nn.PReLU()]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        self.add_block = add_block
        
        ##### original code, if use cosmargin, comment the below lines ####
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num, bias=False)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier
        
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        
        return x

def gem(x, p=2, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):

    def __init__(self, p=1, eps=1e-8):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

class Baseline(nn.Module):
    in_planes = 2048
    reduction = 1
    def __init__(self, cfg,num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        self.tdeep=cfg.MODEL.TDeep

        global BN

        if model_name == 'resnet18':
            self.in_planes = 512
            self.reduction = 2
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2], bn_group=bn_group, bn_var_mode=bn_var_mode)
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3], bn_group=bn_group, bn_var_mode=bn_var_mode)
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3], bn_group=bn_group, bn_var_mode=bn_var_mode)
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3], bn_group=bn_group, bn_var_mode=bn_var_mode)
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3], bn_group=bn_group, bn_var_mode=bn_var_mode)
        elif model_name == 'PSwin':
            self.num_stripes=cfg.MODEL.num_stripes
            if cfg.MODEL.Swin_model==0:
                self.base = PSwinTransformer(depths=[ 2, 2, 6, 2 ],
                    num_heads=[ 3, 6, 12, 24 ],
                    embed_dim=96,
                    model_path=model_path,
                    num_stripes=cfg.MODEL.num_stripes,
                    pre_trained=pretrain_choice,
                    multi_head=cfg.MODEL.DThead_multi)
                if self.tdeep==0 or self.tdeep==1 or self.tdeep==2:
                  self.in_planes = 768
                  self.reduction = cfg.MODEL.nssd_feature_dim
                  self.in_planes2 = int(768/self.num_stripes)
                  self.reduction2 = int(cfg.MODEL.nssd_feature_dim/self.num_stripes)
                elif self.tdeep==3:
                  self.in_planes = 1536
                  self.reduction = cfg.MODEL.nssd_feature_dim
                  self.in_planes2 = int(1536/self.num_stripes)
                  self.reduction2 = int(cfg.MODEL.nssd_feature_dim/self.num_stripes)
                else:
                  pass
            else:
                print("only support swin-tiny (0) in this version!")
        elif model_name == 'Pres50':
            self.num_stripes=cfg.MODEL.num_stripes
            self.base = Presnet(img_size=(256, 256),
                model_path=model_path,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                num_stripes=self.num_stripes,
                last_stride=last_stride,
                embed_dim=256,
                pre_trained=pretrain_choice,
                multi_head=cfg.MODEL.DThead_multi)
            self.in_planes = 2048
            self.reduction = cfg.MODEL.nssd_feature_dim
            self.in_planes2 = int(2048/self.num_stripes)
            self.reduction2 = int(cfg.MODEL.nssd_feature_dim/self.num_stripes)           
        self.model_name=model_name

        if pretrain_choice == 'imagenet':
            if model_name == 'PSwin' or model_name=="Pres50":
                pass
            else:
                self.base.load_param(model_path)
                print('Loading pretrained ImageNet model {}......'.format(model_path))

        self.update_param()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.aap = nn.AdaptiveAvgPool2d(2)

        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.cam=cfg.MODEL.nssd_grad_cam

        if self.neck == 'bnneck':
            if self.model_name=="PSwin" or self.model_name == 'Pres50':
                pass
            else:
                self.bnneck = nn.BatchNorm1d(int(self.in_planes/self.reduction))
                self.bnneck.bias.requires_grad_(False)  # no shift
                self.bnneck.apply(weights_init_kaiming)

                self.classifier = nn.Linear(int(self.in_planes/self.reduction), self.num_classes, bias=False)
                self.classifier.apply(weights_init_classifier)

        if self.model_name=="PSwin" or self.model_name == 'Pres50':
            self.classifier = nn.Linear(int(self.reduction), self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.reduction_layer1 = nn.Linear(int(self.in_planes), int(self.reduction), bias=False)
            self.reduction_layer1.apply(weights_init_classifier)
            self.bnneck1 = nn.BatchNorm1d(int(self.reduction))
            self.bnneck1.bias.requires_grad_(False)
            self.bnneck1.apply(weights_init_kaiming)
            self.bn_layers = nn.ModuleList()
            self.reduction_layers = nn.ModuleList()
            self.cls_layers = nn.ModuleList()
            self.bn_layers.append(self.bnneck1)
            self.reduction_layers.append(self.reduction_layer1)
            self.cls_layers.append(self.classifier)
            for i in range(self.num_stripes):
                self.classifier2 = nn.Linear(int(self.reduction2), self.num_classes, bias=False)
                self.classifier2.apply(weights_init_classifier)
                self.cls_layers.append(self.classifier2)
                self.bnneck2 = nn.BatchNorm1d(int(self.reduction2))
                self.bnneck2.bias.requires_grad_(False)  # no shift
                self.bnneck2.apply(weights_init_kaiming)
                self.bn_layers.append(self.bnneck2)
                self.reduction_layer2 = nn.Linear(int(self.in_planes2), int(self.reduction2), bias=False)
                self.reduction_layer2.apply(weights_init_classifier)
                self.reduction_layers.append(self.reduction_layer2)
        else:
            self.reduction_layer = nn.Sequential(
                nn.BatchNorm2d(self.in_planes),
                nn.ReLU(),
                nn.Conv2d(self.in_planes, int(self.in_planes/self.reduction), kernel_size=1, padding=0, bias=True),
            )
            self.reduction_layer.apply(weights_init_kaiming)
        
        self.unorm = T.Compose(
            [T.Normalize(mean = [ 0., 0., 0. ], std = [ 1./0.229, 1./0.224, 1./0.225 ]), 
             T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])]
              )

    def forward(self, x, epoch=0,writer=None, n_iter=None):

        if self.model_name=="PSwin" or self.model_name=="PVit" or self.model_name == 'Pres50':
            base_feats= self.base(x,self.tdeep)
            global_feat = self.reduction_layers[0](base_feats[0])
            if self.model_name == 'Pres50' and self.tdeep==0:
                all_feats=[global_feat]
            else:
                all_feats=[global_feat]
                local_feat=[]
                for i in range(self.num_stripes):
                    tmp_feat = self.reduction_layers[i+1](base_feats[i+1])
                    all_feats.append(tmp_feat)
        else:
            base_feat = self.base(x)
            reduce_base_feat = self.reduction_layer(base_feat)
            new_base_feat = reduce_base_feat
            global_feat = self.gap(new_base_feat)
            global_feat = global_feat.view(global_feat.shape[0], -1)

        if self.model_name=="PSwin" or  self.model_name == 'Pres50':
            if self.neck == 'no':
                feat = all_feats
            elif self.neck == 'bnneck':
                bn_all_feats=[]
                if self.model_name == 'Pres50' and self.tdeep==0:
                    tmp_feat = self.bn_layers[0](all_feats[0])
                    bn_all_feats.append(tmp_feat)
                else:
                    for i in range(self.num_stripes+1):
                        tmp_feat = self.bn_layers[i](all_feats[i])
                        bn_all_feats.append(tmp_feat)
        else:
            if self.neck == 'no':
                feat = global_feat
            elif self.neck == 'bnneck':
                feat = self.bnneck(global_feat)

        if self.training:
            if self.model_name=="PSwin" or  self.model_name == 'Pres50':
                image_features=torch.cat(all_feats,dim=1)
                all_scores=[]              
                if self.model_name == 'Pres50' and self.tdeep==0:
                    tmp_score = self.cls_layers[0](bn_all_feats[0])
                    all_scores.append(tmp_score)
                else:
                    for i in range(self.num_stripes+1):
                        tmp_score = self.cls_layers[i](bn_all_feats[i])
                        all_scores.append(tmp_score)
                return all_scores,all_feats,image_features
            else:
                cls_score = self.classifier(feat)
                return cls_score, global_feat  # global feature for triplet loss
        else:
            # print("evaluate here")
            if self.neck_feat == 'after':
                # return feat
                if self.model_name=="PSwin" or self.model_name == 'Pres50':
                    if self.model_name == 'Pres50' and self.tdeep==0:
                        return bn_all_feats[0], bn_all_feats[0],bn_all_feats[0]
                    else:
                        tmp1=bn_all_feats[0]+torch.cat(bn_all_feats[1:],dim=1)
                        tmp2=torch.cat(bn_all_feats,dim=1)
                        return bn_all_feats[0], bn_all_feats[0]+torch.cat(bn_all_feats[1:],dim=1),torch.cat(bn_all_feats,dim=1)
            else:
                if self.model_name=="PSwin" or self.model_name == 'Pres50':
                    if self.model_name == 'Pres50' and self.tdeep==0:
                        return all_feats[0], all_feats[0],all_feats[0]
                    else:
                        tmp1=all_feats[0]+torch.cat(all_feats[1:],dim=1)
                        tmp2=torch.cat(all_feats,dim=1)
                        return all_feats[0], all_feats[0]+torch.cat(all_feats[1:],dim=1),torch.cat(all_feats,dim=1)
                else:
                    return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['state_dict']
        for key in param_dict:
            key_cut = key[7:]
            if 'classifier' in key_cut:
                continue
            self.state_dict()[key_cut].copy_(param_dict[key])

    def load_param_swin(self, trained_path):
        param_dict=torch.load(trained_path, 'cpu')['model']
        for key in param_dict:
            new_key = "base."+key
            if 'head' in key:
                continue
            try:
              self.state_dict()[new_key].copy_(param_dict[key])
            except:
              print(key)
              exit()

    def update_param(self):
        pass




