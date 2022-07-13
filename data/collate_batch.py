# encoding: utf-8
"""
@author:  qianwen
@contact: qianwen2018@ia.ac.cn
"""

import torch


def train_collate_fn(batch):
    imgs, pids, camids, tids, image_ids, score_maps, rois = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    tids = torch.tensor(tids, dtype=torch.int64)
    # if score_maps[0] is not None:
    #     score_maps = torch.stack(score_maps, 0)
    # else: score_maps = None
    new_score_maps = None
    if score_maps[0] is not None:
        # new_score_maps = defaultdict(list)
        new_score_maps={}
        score_keys = list(score_maps[0].keys())
        for score_map in score_maps:
            for key in score_keys:
                if not key in new_score_maps.keys():
                    new_score_maps[key]=[]
                new_score_maps[key].append(torch.from_numpy(score_map[key]))
        for key in new_score_maps:
            new_score_maps[key] = torch.stack(new_score_maps[key], 0)
    if rois[0] is not None: rois = torch.stack(rois, 0)
    else: rois = None
    return torch.stack(imgs, dim=0), pids, camids, tids, image_ids, rois

def val_collate_fn(batch):
    imgs, pids, camids, tids, image_ids, score_maps, rois = zip(*batch)
    # print("here")
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    tids = torch.tensor(tids, dtype=torch.int64)
    # if score_maps[0] is not None:
    #     score_maps = torch.stack(score_maps, 0)
    # else: score_maps = None
    # print("here2")
    new_score_maps = None
    if score_maps[0] is not None:
        # print(list(score_maps[0].keys()))
        # exit()
        # new_score_maps = defaultdict(list)
        new_score_maps={}
        score_keys = list(score_maps[0].keys())
        # print(score_keys)
        # exit()
        for score_map in score_maps:
            for key in score_map:
                if not key in new_score_maps.keys():
                    new_score_maps[key]=[]
                new_score_maps[key].append(torch.from_numpy(score_map[key]))
        for key in new_score_maps:
            new_score_maps[key] = torch.stack(new_score_maps[key], 0)
    if rois[0] is not None: rois = torch.stack(rois, 0)
    else: rois = None
    # print("here 3")
    return torch.stack(imgs, dim=0), pids, camids, tids, image_ids, rois


def train_collate_fn_dali(batch):
    imgs, pids, camids, tids, image_ids, score_maps, rois = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    tids = torch.tensor(tids, dtype=torch.int64)
    if score_maps[0] is not None:
        score_maps = torch.stack(score_maps, 0)
    else: score_maps = None
    if rois[0] is not None: rois = torch.stack(rois, 0)
    else: rois = None
    return imgs, pids, camids, tids, image_ids, rois

def val_collate_fn_dali(batch):
    imgs, pids, camids, tids, image_ids, score_maps, rois = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    tids = torch.tensor(tids, dtype=torch.int64)
    if score_maps[0] is not None:
        score_maps = torch.stack(score_maps, 0)
    else: score_maps = None
    if rois[0] is not None: rois = torch.stack(rois, 0)
    else: rois = None
    return imgs, pids, camids, tids, image_ids, rois