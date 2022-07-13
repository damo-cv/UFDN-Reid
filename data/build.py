# encoding: utf-8
"""
@author:  qianwen
@contact: qianwen2018@ia.ac.cn
"""

import torch
from torch.utils.data import Dataset, DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
import torch.distributed as dist
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid
from .samplers import RandomIdentitySampler_DDP
from .transforms import build_transforms
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(),4)


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, data_filter=cfg.DATASETS.filter,offline_path=cfg.MODEL.OFFLINE_PATH)
    val_set = ImageDataset(dataset.query + dataset.gallery, transform=val_transforms, size=cfg.INPUT.SIZE_TEST, is_train=False)
    # val_set = ImageDataset(dataset.train, transform=val_transforms, size=cfg.INPUT.SIZE_TEST, is_train=False)
    train_set = ImageDataset(dataset.train, transform=train_transforms, size=cfg.INPUT.SIZE_TRAIN, is_train=True)
    num_classes = dataset.num_train_pids
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
    # data_sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
    data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
    train_loader = DataLoaderX(
        # train_set, 
        # batch_size=cfg.SOLVER.IMS_PER_BATCH,
        # sampler=RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        # num_workers=num_workers, 
        # collate_fn=train_collate_fn
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=train_collate_fn,
        pin_memory=False,
    )

    val_loader = DataLoaderX(
        val_set, 
        batch_size=cfg.TEST.IMS_PER_BATCH, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.train),len(dataset.query), len(dataset.gallery), num_classes