# encoding: utf-8
# ns_structure loss represents no-standard structure decouple loss
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy import spatial

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    y = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return y


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = torch.addmm(dist, x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def euclidean_cosine_dist(x, y,dnum):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    dismats=[]
    for i in range(dnum):
        distmat = (1. - torch.matmul(x[:,i,:], y[:,i,:].t())) / 2.
        distmat = distmat.clamp(min=1e-12)
        dismats.append(distmat)
    dismats=torch.cat(dismats,1)
    dismats=torch.sum(dismats.view(n,n,-1),-1)
    return dismats

def eye_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [n, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    n = x.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(n, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, n).t()
    dist = xx + yy
    dist = torch.addmm(dist, x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt().cpu().detach().numpy()  # for numerical stability
    dist = torch.from_numpy(dist.diagonal())

    return dist

def inner_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    distmat = (1. - torch.matmul(x, y.t()))/2.
    # distmat = distmat.clamp(min=1e-12)
    return distmat

def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    x = normalize(x, axis=1)
    y = normalize(y, axis=1)
    distmat = (1. - torch.matmul(x, y.t())) / 2.
    distmat = distmat.clamp(min=1e-12)
    return distmat

def cosine_dist_orth(x, y):
    x = normalize(x, axis=1)
    y = normalize(y, axis=1)
    distmat = (1. - torch.matmul(x, y.t())) *100
    distmat = distmat.clamp(min=1e-12)
    return distmat

def cosine(x):
    x = normalize(x, axis=1)
    distmat=torch.matmul(x, x.t())
    return distmat

class NSSD_loss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self,num_stripes,feat_dim,cluster_center_weight=[0.9,0.1],margin=0,weight_dict=None):
        self.num_stripes=num_stripes
        self.feat_dim=feat_dim
        assert feat_dim%num_stripes==0
        # self.cluster_center=torch.zeros([num_stripes,int(feat_dim/num_stripes)])
        self.cluster_center_weight=cluster_center_weight
        self.margin=margin
        if not margin==0:
            print("MarginRankingLoss",margin)
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
            # exit()
        else:
            print("SoftMarginLoss")
            self.ranking_loss = nn.SoftMarginLoss()
        if weight_dict==None:
            self.weight_dict={}
            self.cluster_center=torch.zeros([num_stripes,int(feat_dim/num_stripes)]).cuda()
        else:
            self.weight_dict=weight_dict
            self.cluster_center=weight_dict['cluster_center'].cuda()

    def __call__(self, global_feat,iters,num_iter,epoch=0):
        N,L=global_feat.size()
        global_feat=global_feat.view(N,self.num_stripes,-1)
        global_feat=normalize(global_feat)
        loss1=torch.zeros(1).cuda()
        # print(global_feat.size())
        for i in range(N):
            # tmp_mat=cosine_dist(global_feat[i], global_feat[i])
            loss1=loss1+torch.sum(torch.abs(cosine(global_feat[i])-torch.eye(self.num_stripes).cuda()))/N/self.num_stripes/self.num_stripes
            # loss1=loss1+torch.sum(torch.abs(torch.ones([self.num_stripes,self.num_stripes]).cuda()-cosine_dist(global_feat[i], global_feat[i])-torch.eye(self.num_stripes).cuda()))/N

        loss2=torch.zeros(1).cuda()
        if torch.sum(self.cluster_center)==0:
            self.cluster_center=torch.sum(global_feat,0)/N
        for i in range(N):
            tmp_mat=cosine_dist(global_feat[i], self.cluster_center.detach())
            # print("tmp_mat",tmp_mat)
            diag = torch.diag(tmp_mat)
            # print("diag",diag)
            # a_diag = torch.diag_embed(diag)
            a_diag=torch.eye(self.num_stripes).cuda()
            # print("a_diag",a_diag)
            tmp_mat_max=torch.min(tmp_mat+a_diag,1)[0]
            # print("tmp_mat_max",tmp_mat_max)
            y = tmp_mat_max.new().resize_as_(tmp_mat_max).fill_(1)
            if not self.margin==0:
                loss2 =loss2+self.ranking_loss(tmp_mat_max,diag, y)/N
            else:
                loss2 =loss2+self.ranking_loss(tmp_mat_max-diag, y)/N
            # print(loss2)
        # exit()
        self.cluster_center=self.cluster_center_weight[0]*self.cluster_center+self.cluster_center_weight[1]*torch.sum(global_feat,0)/N
        self.weight_dict['cluster_center']=self.cluster_center
        return loss1+loss2,self.weight_dict


class NSSD_loss_2(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self,num_stripes,work_epoch,feat_dim,cluster_center_weight=[0.9,0.1],cluster_distance_weight=[0.9,0.1],weight_dict=None):
        self.num_stripes=num_stripes
        self.work_epoch=work_epoch
        self.feat_dim=feat_dim
        assert feat_dim%num_stripes==0
        # self.cluster_center=torch.zeros([num_stripes,int(feat_dim/num_stripes)])
        self.cluster_center_weight=cluster_center_weight
        self.cluster_distance_weight=cluster_distance_weight
        self.ranking_loss = nn.SoftMarginLoss()
        if weight_dict==None:
            self.weight_dict={}
            self.cluster_center=torch.zeros([num_stripes,int(feat_dim/num_stripes)]).cuda()
            self.cluster_avg_dist=torch.zeros([num_stripes]).cuda()
            self.cluster_avg_dist_old=torch.zeros([num_stripes]).cuda()
        else:
            self.weight_dict=weight_dict
            self.cluster_center=weight_dict['cluster_center'].cuda()
            self.cluster_avg_dist=weight_dict['cluster_avg_dist'].cuda()
            self.cluster_avg_dist_old=weight_dict['cluster_avg_dist_old'].cuda()       


    def __call__(self, global_feat,iters,num_iter,epoch=0):
        N,L=global_feat.size()
        global_feat=global_feat.view(N,self.num_stripes,-1)
        global_feat=normalize(global_feat)
        loss1=torch.zeros(1).cuda()
        # print(global_feat.size())
        for i in range(N):
            # tmp_mat=cosine_dist(global_feat[i], global_feat[i])
            loss1=loss1+torch.sum(torch.abs(cosine(global_feat[i])-torch.eye(self.num_stripes).cuda()))/N
            # loss1=loss1+torch.sum(torch.abs(cosine_dist(global_feat[i], global_feat[i])-torch.eye(self.num_stripes).cuda()))/N/self.num_stripes/self.num_stripes
        # return loss1

        loss2=torch.zeros(1).cuda()
        tmp_avg_dist=torch.zeros([self.num_stripes]).cuda()
        if torch.sum(self.cluster_center)==0:
            self.cluster_center=torch.sum(global_feat,0)/N
            # print(self.cluster_center.size())
        else:
            for i in range(N):
                # print(i)
                if epoch<self.work_epoch:
                    tmp_mat=cosine_dist(global_feat[i], self.cluster_center.detach())
                    diag = torch.diag(tmp_mat)
                    a_diag = torch.eye(self.num_stripes).cuda()
                    tmp_mat_max=torch.min(tmp_mat+a_diag,1)[0]
                    y = tmp_mat_max.new().resize_as_(tmp_mat_max).fill_(1)
                    loss2 =loss2+self.ranking_loss(tmp_mat_max-diag, y)
                    if epoch==self.work_epoch-1:
                        tmp_avg_dist=tmp_avg_dist+diag/N

                else:
                    tmp_mat=cosine_dist(global_feat[i], self.cluster_center.detach())
                    diag = torch.diag(tmp_mat)
                    a_diag = torch.eye(self.num_stripes).cuda()
                    tmp_mat_max=torch.min(tmp_mat+a_diag,1)[0]
                    mask=torch.tensor([1 if value<2*center_value else 0 for value,center_value in zip(diag,self.cluster_avg_dist_old)]).cuda()
                    # print("a_diag_max 0",a_diag_max)
                    a_diag_max=diag*mask
                    # print("a_diag_max 1",a_diag_max)
                    y = a_diag_max.new().resize_as_(a_diag_max).fill_(1)
                    if torch.sum(mask)>0:
                        # loss2 =loss2+self.ranking_loss(tmp_mat_max-diag, y)/torch.sum(mask)
                        loss2 =loss2+self.ranking_loss(tmp_mat_max-diag, y)
                    else:
                        loss2 =loss2
                    if epoch==self.work_epoch-1:
                        tmp_avg_dist=tmp_avg_dist+diag/N
            if epoch>self.work_epoch-2:
                self.cluster_avg_dist=self.cluster_distance_weight[0]*self.cluster_avg_dist+self.cluster_distance_weight[1]*tmp_avg_dist
            # elif epoch==self.work_epoch-1:
            #     self.cluster_avg_dist=tmp_avg_dist
            self.cluster_center=self.cluster_center_weight[0]*self.cluster_center+self.cluster_center_weight[1]*torch.sum(global_feat,0)/N
            # print("finished")
        if epoch>self.work_epoch-2 and iters==num_iter:
            self.cluster_avg_dist_old=self.cluster_avg_dist
        self.weight_dict['cluster_center']=self.cluster_center
        self.weight_dict['cluster_avg_dist']=self.cluster_avg_dist
        self.weight_dict['cluster_avg_dist_old']=self.cluster_avg_dist_old  
        return loss1+loss2,self.weight_dict

class NSSD_loss_3(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self,num_stripes,feat_dim,cluster_center_weight=[0.9,0.1],margin=None,weight_dict=None):
        print("NSSD_loss_3")
        self.num_stripes=num_stripes
        self.feat_dim=feat_dim
        assert feat_dim%num_stripes==0
        # self.cluster_center=torch.zeros([num_stripes,int(feat_dim/num_stripes)])
        self.cluster_center_weight=cluster_center_weight
        self.margin=margin
        if not margin==0:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()
        if weight_dict==None:
            self.weight_dict={}
            self.cluster_center=torch.zeros([num_stripes,int(feat_dim/num_stripes)])
        else:
            self.weight_dict=weight_dict
            self.cluster_center=weight_dict['cluster_center'].cuda()

    def __call__(self, global_feat,iters,num_iter,epoch=0):
        N,L=global_feat.size()
        global_feat=global_feat.view(N,self.num_stripes,-1)
        global_feat=normalize(global_feat)
        loss1=torch.zeros(1).cuda()
        for i in range(N):
            loss1=loss1+torch.sum(torch.abs(cosine(global_feat[i])-torch.eye(self.num_stripes).cuda()))/N

        loss2=torch.zeros(1).cuda()
        if torch.sum(self.cluster_center)==0:
            self.cluster_center=torch.sum(global_feat,0)/N
        for i in range(N):
            tmp_mat=cosine_dist(global_feat[i], self.cluster_center.detach())
            diag = torch.diag(tmp_mat)
            a_diag=torch.eye(self.num_stripes).cuda()
            tmp_mat_max=torch.min(tmp_mat+a_diag,1)[0]
            y = tmp_mat_max.new().resize_as_(tmp_mat_max).fill_(1)
            if not self.margin==0:
                loss2 =loss2+self.ranking_loss(tmp_mat_max,diag, y)
            else:
                loss2 =loss2+self.ranking_loss(tmp_mat_max-diag, y)
        self.cluster_center=self.cluster_center_weight[0]*self.cluster_center+self.cluster_center_weight[1]*torch.sum(global_feat,0)/N
        self.weight_dict['cluster_center']=self.cluster_center
        return loss1+loss2,self.weight_dict


if __name__ == "__main__":
    np.random.seed(123)
    
    feat = torch.FloatTensor(np.random.uniform(0, 1, (6, 2048)))
    feat2 = torch.FloatTensor(np.random.uniform(0, 1, (6, 2048)))
    target = torch.LongTensor([0,1,1,2,0,2])
    
    nssd_loss = NSSD_loss(8,20,2048)
    nssd_loss(feat)
    nssd_loss(feat2)

    # print(score.size(), target.size())
    # print(triplet1(score, target))