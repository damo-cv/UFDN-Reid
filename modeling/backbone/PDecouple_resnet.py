import math
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Mlp as Mlp_old

from .resnet import ResNet, BasicBlock, Bottleneck
import copy

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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA 
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    
    def forward(self, x ):
        
        B, N, C = x.shape
        # print(x.size(),B, 1, self.num_heads, C // self.num_heads)
        # exit()
        q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        
        return x_cls  

class LayerScale_Block_CA(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Class_Attention,
                 Mlp_block=Mlp,init_values=1e-4,num_stripes=1):
        super().__init__()
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn2 = Attention_block(
            int(dim/num_stripes), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.num_stripes=num_stripes
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp_block(in_features=int(dim/num_stripes), hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_list=nn.ModuleList()
        self.gamma_list=[]
        for i in range(2):
            self.gamma = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True).cuda()
            self.norm_list.append(norm_layer(dim))
            self.gamma_list.append(self.gamma)
        for i in range(num_stripes*2):
            self.gamma = nn.Parameter(init_values * torch.ones((int(dim/num_stripes))),requires_grad=True).cuda()
            self.norm_list.append(norm_layer(int(dim/num_stripes)))
            self.gamma_list.append(self.gamma)
    
    def forward(self, global_feat,local_feats, cls_tokens):
        u1 = torch.cat((cls_tokens[0],global_feat),dim=1)
        u_feats=[u1]
        for i in range(len(local_feats)):
            u_tmp=torch.cat((cls_tokens[i+1],local_feats[i]),dim=1)
            u_feats.append(u_tmp)  
        x_cls1 = cls_tokens[0] + self.drop_path(self.gamma_list[0] * self.attn(self.norm_list[0](u1)))
        x_cls1 = x_cls1 + self.drop_path(self.gamma_list[1] * self.mlp(self.norm_list[1](x_cls1)))
        x_cls_list=[x_cls1]
        for i in range(self.num_stripes):
            x_cls2 = cls_tokens[i+1] + self.drop_path(self.gamma_list[2+i*2] * self.attn2(self.norm_list[2+i*2](u_feats[i+1])))
            x_cls2 = x_cls2 + self.drop_path(self.gamma_list[3+i*2] * self.mlp2(self.norm_list[3+i*2](x_cls2)))
            x_cls_list.append(x_cls2)
        return x_cls_list



class Presnet(nn.Module):
    def __init__(self, img_size=(224, 224),model_path="",embed_dim=96,depth_token_only=2, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.5,stride_size=16,
                 norm_layer=nn.LayerNorm,num_stripes=2,last_stride=1,pre_trained=None,multi_head=0,**kwargs):
        super().__init__()
        # self.vit_base=vit_base_patch16_224_TransReID(img_size=img_size,stride_size=stride_size,model_path=model_path,**kwargs)
        self.base = ResNet(last_stride=last_stride,
                   block=Bottleneck,
                   layers=[3, 4, 6, 3])
        self.num_stripes=num_stripes
        self.cls_tokens_list = []
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim*8))
        self.cls_tokens_list.append(self.cls_token)
        for i in range(num_stripes):
            self.cls_token = nn.Parameter(torch.zeros(1, 1, int(embed_dim*8/num_stripes)))
            self.cls_tokens_list.append(self.cls_token)
        self.num_features=2048
        self.norm1 = norm_layer(self.num_features)
        self.norm2 = norm_layer(self.num_features)
        self.multi_head=multi_head

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.blocks_token_only = nn.ModuleList([
            LayerScale_Block_CA(
                dim=embed_dim*8, num_heads=8, mlp_ratio=4.0, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=nn.GELU,Attention_block=Class_Attention,
                Mlp_block=Mlp_old,init_values=1e-4,num_stripes=num_stripes)
            for i in range(depth_token_only)])


        self.reduction_layer = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=1, padding=0, bias=True),
        )
        self.reduction_layer.apply(weights_init_kaiming)
        self.reduction_layer2 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=1, padding=0, bias=True),
        )
        self.reduction_layer2.apply(weights_init_kaiming)

        self.apply(self._init_weights)
        if model_path and pre_trained=="imagenet":
            self.base.load_param(model_path)
        self.backbone = nn.Sequential(
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
        )
        self.layer3 = copy.deepcopy(self.base.layer3)
        self.layer4 = copy.deepcopy(self.base.layer4)
        self.layer3_2 = copy.deepcopy(self.base.layer3)
        self.layer4_2 = copy.deepcopy(self.base.layer4)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x,tdeep=1):
        B = x.shape[0]
        cls_tokens_list=[]
        for i in range(len(self.cls_tokens_list)):
            cls_token_tmp = self.cls_tokens_list[i].expand(B, -1, -1).cuda()
            cls_tokens_list.append(cls_token_tmp)

        x=self.backbone(x)
        global_feat=self.layer3(x)
        global_feat=self.layer4(global_feat)
        feat_dim=global_feat.size(1)


        local_feat=self.layer3_2(x)
        local_feat=self.layer4_2(local_feat)

        global_feat=global_feat.view(B,feat_dim,-1).transpose(1, 2).contiguous()
        local_feat=local_feat.view(B,feat_dim,-1).transpose(1, 2).contiguous()

        if tdeep==0:
            global_feat = self.avgpool(global_feat.transpose(1, 2))  # B C 1
            global_feat = torch.flatten(global_feat, 1)        
            return [global_feat]


        global_feat = self.norm1(global_feat)
        local_feat = self.norm2(local_feat)
        local_feat_chunk = local_feat.chunk(self.num_stripes, dim=2)

        for i , blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(global_feat,local_feat_chunk,cls_tokens_list)
        global_feat = self.avgpool(global_feat.transpose(1, 2))  # B C 1
        global_feat = torch.flatten(global_feat, 1)
        final_feats=[global_feat]
        for i in range(self.num_stripes):
            x1 = self.avgpool(local_feat_chunk[i].transpose(1, 2))  # B C 1
            x1 = torch.flatten(x1, 1)
            final_feats.append(x1)


        return_feats=[]
        for i in range(self.num_stripes+1):
            tmp_feat=final_feats[i]+cls_tokens[i][:,0,:]
            # print(tmp_feat.size())
            return_feats.append(tmp_feat)
        # exit()
        return return_feats





