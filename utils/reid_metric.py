# encoding: utf-8


import numpy as np
from scipy.spatial.distance import cdist

import torch
from ignite.metrics import Metric

# from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking, re_ranking_numpy

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=100):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx]/ (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    # mAP = pytrec_mAP(distmat, q_pids, g_pids, q_camids, g_camids, topk=max_rank)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

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
    # print(dismats.size())
    # exit()

    # xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    # dist = xx + yy
    # dist = torch.addmm(dist, x, y.t(), beta=1, alpha=-2)
    # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dismats


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes', IF_mINP=False, NUM_DECOUPLE=1):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.IF_mINP = IF_mINP
        self.NUM_DECOUPLE=NUM_DECOUPLE

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.tids = []

    def update(self, output):
        feat, pid, camid, tid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.tids.extend(np.asarray(tid))
        self.unique_tids = list(set(self.tids))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        gallery_tids = np.asarray(self.tids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]

        if self.NUM_DECOUPLE==1:
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat = torch.addmm(distmat, qf, gf.t(), beta=1, alpha=-2)
            distmat = distmat.cpu().numpy()
            # print(distmat)
        else:
            qf=qf.view(m,self.NUM_DECOUPLE,-1)
            qf = normalize(qf, axis=-1)
            gf=gf.view(n,self.NUM_DECOUPLE,-1)
            gf = normalize(gf, axis=-1)
            distmat=[]
            for i in range(self.NUM_DECOUPLE):
                distmat_tmp = (1-torch.matmul(qf[:,i,:], gf[:,i,:].t()))/2.
                distmat_tmp = distmat_tmp.clamp(min=1e-12)
                distmat_tmp = distmat_tmp.view(m,n,1)
                distmat.append(distmat_tmp)
            distmat=torch.cat(distmat,2)
            # print(distmat.size())
            distmat=torch.sum(distmat.view(m,n,-1),-1)/self.NUM_DECOUPLE
            # print(distmat)
            # exit()


        # distmat = re_ranking(qf, gf, 30, 6, 0.3, local_distmat=None, only_local=False)

        # distmat = (1. - torch.matmul(qf, gf.t())) / 2.
        # distmat = distmat.clamp(min=1e-12)
        
        cmc, mAP, mINP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=self.max_rank)
        if not self.IF_mINP:
            return cmc, mAP
        else:
            return cmc, mAP, mINP

def invweight(dist, num=1., const=0.000001):
    return num / (dist + const)

def subweight(dist, const=1.):
    return const-dist if dist <= const else 0.

def gaussian(dist, sigma=10.):
    return math.e**(-dist**2/2/sigma**2)

def norm_weight(embedding):
    embedding = np.asarray([invweight(l) for l in embedding])
    embedding = embedding / embedding.sum()
    return embedding

class Track_R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes', IF_mINP=False):
        super(Track_R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.IF_mINP = IF_mINP

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.tids = []

    def update(self, output):
        feat, pid, camid, tid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.tids.extend(np.asarray(tid))
        self.unique_tids = list(set(self.tids))

    def track_ranking(self, qf, gf, gallery_tids, unique_tids):
        origin_dist = cdist(qf, gf)
        m, n = qf.shape[0], gf.shape[0]
        feature_dim = qf.shape[1]
        gallery_tids = np.asarray(gallery_tids)
        unique_tids = np.asarray(unique_tids)
        track_gf = np.zeros((len(unique_tids), feature_dim))
        dist = np.zeros((m, n))
        gf_tids = sorted(list(set(gallery_tids)))
        for i, tid in enumerate(gf_tids):
            track_gf[i, :] = np.mean(gf[gallery_tids == tid, :] , axis=0)
        track_dist = cdist(qf, track_gf)
        track_dist = re_ranking_numpy(qf, track_gf, k1=8, k2=3, lambda_value=0.3)
        # track_dist = re_ranking_numpy(qf, track_gf, k1=10, k2=3, lambda_value=0.3)
        for i, tid in enumerate(gf_tids):
            dist[:, gallery_tids == tid] = track_dist[:, i:(i+1)]
        for i in range(m):
            for tid in gf_tids:
                min_value = np.min(origin_dist[i][gallery_tids == tid])
                min_index = np.where(origin_dist[i] == min_value)
                min_value = dist[i][min_index[0][0]]
                dist[i][gallery_tids == tid] = min_value + 0.000001
                dist[i][min_index] = min_value
        return dist

    def track_reranking_weight_feat(self, qf, gf, gallery_tids, unique_tids):
        origin_q_g_dist = cdist(qf, gf)

        m, n = qf.shape[0], gf.shape[0]
        feature_dim = qf.shape[1]
        gallery_tids = np.asarray(gallery_tids)
        unique_tids = np.asarray(unique_tids)
        track_gf = np.zeros((len(unique_tids), feature_dim))
        dist = np.zeros((m, n))
        gf_tids = sorted(list(set(gallery_tids)))

        for i, tid in enumerate(gf_tids):
            weight = origin_q_g_dist[:, gallery_tids == tid].mean(axis=0)
            weight = norm_weight(weight)
            track_gf[i, :] = (gf[gallery_tids == tid] * weight[:, np.newaxis]).sum(axis=0)

        track_dist = re_ranking_numpy(qf, track_gf, k1=8, k2=3, lambda_value=0.3)
        for i, tid in enumerate(gf_tids):
            dist[:, gallery_tids == tid] = track_dist[:, i:(i+1)]
        return dist

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        gallery_tids = np.asarray(self.tids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]

        qf = qf.cpu().numpy()
        gf = gf.cpu().numpy()
        distmat = self.track_ranking(qf, gf, gallery_tids, self.unique_tids)
        # distmat = self.track_reranking_weight_feat(qf, gf, gallery_tids, self.unique_tids)
        
        cmc, mAP, mINP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=self.max_rank)

        if not self.IF_mINP:
            return cmc, mAP
        else:
            return cmc, mAP, mINP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes', IF_mINP=False):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.IF_mINP = IF_mINP

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.tids = []

    def update(self, output):
        feat, pid, camid, tid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.tids.extend(np.asarray(tid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=30, k2=6, lambda_value=0.3)
        cmc, mAP, mINP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=self.max_rank)
        if not self.IF_mINP:
            return cmc, mAP
        else:
            return cmc, mAP, mINP