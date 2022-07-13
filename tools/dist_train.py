# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import argparse
import os
from os.path import join as opj
import sys
import time
import numpy as np
import collections
from glob import glob
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
# from data import make_data_loader_dali
# from modeling import build_model_config
from modeling import build_model
from layers import make_loss
from solver import *
from solver.scheduler_factory import create_scheduler

from utils.logger import setup_logger
from utils.reid_metric import R1_mAP, Track_R1_mAP
from utils.analyse import count_params, count_flops
# from layers import CrossEntropySoftLabel
import torch.distributed as dist


# import spring.linklink as link
# from dist.distributed_utils import dist_init, reduce_gradients, DistModule, AllGather, AllReduce

from tensorboardX import SummaryWriter

# from dirichlet import dirichlet, set_logger_path, set_tb_logger, dump_model
# from dirichlet.convert import ppl, trt
import yaml

from train import *

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_optimizer(cfg, model,n_iter_per_epoch, paras):

    # if cfg.MODEL.IF_METRIC == 'yes' or cfg.MODEL.IF_BFE == 'yes' or cfg.MODEL.IF_ROI == 'yes':
    if cfg.MODEL.IF_METRIC == 'yes' or cfg.MODEL.IF_BFE == 'yes':
        optimizer = make_optimizer_two_stream(cfg, model)
    elif cfg.MODEL.IF_DOMAIN == 'yes':
        optimizer = make_optimizer_domain(cfg, model)
    else:
        optimizer = make_optimizer(cfg, model, paras)

    if cfg.SOLVER.scheduler_mode == "linear":
        print("linear lr")
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    elif cfg.SOLVER.scheduler_mode=="Cos":
        print("cosine lr")
        scheduler=create_scheduler(cfg, optimizer)
    return optimizer, scheduler

def train(cfg, logger, local_rank):
    # prepare dataset
    # DEVICE_ID=[int(x) for x in cfg.MODEL.DEVICE_ID.split(',')]
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, val_loader, num_train,num_query, num_gallery, num_classes = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    n_iter_per_epoch=int(num_train/cfg.SOLVER.IMS_PER_BATCH/4)
    # model.cuda()

    loss_func = make_loss(cfg, num_classes)

    start_epoch = cfg.MODEL.STATE_EPOCH

    # if cfg.MODEL.TFreeze:
    #     for key in model.state_dict().keys():
    #         if "base2" in key:
    #             model.state_dict()[key].requires_grad = False
    #         if "reduction_layer2" in key:
    #             model.state_dict()[key].requires_grad = False
    #         if "bnneck2" in key:
    #             model.state_dict()[key].requires_grad = False
    #         if "classifier2" in key:
    #             model.state_dict()[key].requires_grad = False
    if cfg.MODEL.IF_MUL=="yes":
        # log_var_a = torch.zeros((1,))[0].cuda()
        # log_var_b = torch.zeros((1,))[0].cuda()
        log_var_a = torch.ones((1,))[0].cuda()
        log_var_b = torch.ones((1,))[0].cuda()
        log_var_c = torch.ones((1,))[0].cuda()
        log_var_a.requires_grad = True
        log_var_b.requires_grad = True
        log_var_c.requires_grad = True
        paras=[log_var_a,log_var_b,log_var_c]
    else:
        paras=[]
    nssd_weight_dict=None


    # Add for using self trained model
    if cfg.MODEL.PRETRAIN_CHOICE == 'self':
        print('loading pre-train model {}'.format(cfg.MODEL.PRETRAIN_PATH))
        predict_dict = torch.load(cfg.MODEL.PRETRAIN_PATH, 'cpu')
        if 'state_dict' in predict_dict:
            predict_dict = predict_dict['state_dict']
        model_dict = model.state_dict()

        new_dict = collections.OrderedDict()
        for k, v in predict_dict.items():
            if k[7:] in model.state_dict().keys() and v.size() == model.state_dict()[k[7:]].size():
                if 'base' not in k: continue
                new_dict[k[7:]] = v
        # if rank == 0:
        print('loading params {}'.format(new_dict.keys()))
        for k, v in model_dict.items():
            if k not in new_dict.keys():
                new_dict[k] = v
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        model.update_param()
        optimizer, scheduler = get_optimizer(cfg, model, n_iter_per_epoch)
    elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        optimizer, scheduler = get_optimizer(cfg, model, n_iter_per_epoch, paras)
    elif cfg.MODEL.PRETRAIN_CHOICE == 'state':
        print('loading state model {}'.format(cfg.MODEL.PRETRAIN_PATH))
        state_metas = torch.load(cfg.MODEL.PRETRAIN_PATH, 'cpu')
        nssd_weight_dict=state_metas['nssd_weight'] if 'nssd_weight' in state_metas else None
        start_epoch = state_metas['last_epoch'] if 'last_epoch' in state_metas else 0
        # if cfg.MODEL.PMode:
        #     start_epoch=0
        # print(start_epoch)
        # exit()
        state_optimizer = state_metas['optimizer']
        predict_dict = state_metas['state_dict']
        # print(state_optimizer)
        # exit()
        model_dict = model.state_dict()
        new_dict = collections.OrderedDict()

        for k, v in predict_dict.items():
            if k in model.state_dict().keys() and v.size() == model.state_dict()[k].size():
                new_dict[k] = v.cuda()
        print('loading params {}'.format(new_dict.keys()))
        for k, v in model_dict.items():
            if k not in new_dict.keys():
                new_dict[k] = v.cuda()
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        if cfg.SOLVER.scheduler_mode == "linear":
            optimizer, _ = get_optimizer(cfg, model, n_iter_per_epoch, paras)
        else:
            optimizer, scheduler = get_optimizer(cfg, model, n_iter_per_epoch, paras)

        optimizer.load_state_dict(state_optimizer)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        if cfg.SOLVER.scheduler_mode == "linear":
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
    elif cfg.MODEL.PRETRAIN_CHOICE == 'qw':
        print('loading state model {}'.format(cfg.MODEL.PRETRAIN_PATH))
        state_metas = torch.load(cfg.MODEL.PRETRAIN_PATH, 'cpu')
        start_epoch = 0
        predict_dict = state_metas['state_dict']
        model_dict = model.state_dict()
        new_dict = collections.OrderedDict()
        for k, v in predict_dict.items():
            if k[7:] in model.state_dict().keys() and v.size() == model.state_dict()[k[7:]].size():
                new_dict[k[7:]] = v
        print('loading params {}'.format(new_dict.keys()))
        for k, v in model_dict.items():
            if k not in new_dict.keys():
                new_dict[k] = v
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        optimizer, scheduler = get_optimizer(cfg, model, n_iter_per_epoch)
    elif cfg.MODEL.PRETRAIN_CHOICE == 'evaluate':
        state_metas = torch.load(cfg.MODEL.PRETRAIN_PATH, 'cpu')
        state_optimizer = state_metas['optimizer']
        predict_dict = state_metas['state_dict']
        model_dict = model.state_dict()
        new_dict = collections.OrderedDict()
        # print(predict_dict.keys())
        # print(model_dict.keys())
        # exit()
        for k, v in predict_dict.items():
            if k[7:] in model.state_dict().keys() and v.size() == model.state_dict()[k[7:]].size():
                new_dict[k[7:]] = v
        print('loading params {}'.format(new_dict.keys()))
        for k, v in model_dict.items():
            if k not in new_dict.keys():
                new_dict[k] = v
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        optimizer, scheduler = get_optimizer(cfg, model, n_iter_per_epoch,paras)
    else:
        print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

    # if cfg.QCONFIG != '' and os.path.exists(cfg.QCONFIG):
    #     print('begin quantization...')
    #     model.cpu()
    #     dummy_inputs = (torch.randn(1, 3, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]), torch.Tensor([cfg.SOLVER.IMS_PER_BATCH]).int())
    #     with open(cfg.QCONFIG) as f:
    #         qconfig = yaml.load(f)
    #     drcl_scheduler = dirichlet(model, dummy_inputs, qconfig)
    #     set_logger_path(cfg.OUTPUT_DIR + '/dirichlet.log')
    #     model.cuda()
    # else: 
    #     drcl_scheduler = None

    # model.cuda()
    do_train_config(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,      # modify for using self trained model
        loss_func,
        num_query,
        num_gallery,
        start_epoch,     # add for using self trained model
        logger,
        local_rank,
        paras,
        nssd_weight_dict,
        num_classes=num_classes
    )
    # if dist.get_rank() == 0: writer.close()

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # rank, world_size = dist_init()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # if rank==0:
    exists_models=glob(opj(cfg.OUTPUT_DIR,"*model.pth"))
    if exists_models:
        state_metas = torch.load(exists_models[0], 'cpu')
        start_epoch = state_metas['last_epoch']
    else:
        start_epoch = 0
    # # print(cfg.OUTPUT_DIR)

    if exists_models and start_epoch==120:
        cfg.MODEL.PRETRAIN_CHOICE = 'evaluate'
        new_models=glob(opj(cfg.OUTPUT_DIR,"*model_120.pth"))
        cfg.MODEL.PRETRAIN_PATH = new_models[0]
    elif exists_models and start_epoch>10 and not cfg.MODEL.PRETRAIN_CHOICE == 'evaluate':
        cfg.MODEL.PRETRAIN_CHOICE = 'state'
        cfg.MODEL.PRETRAIN_PATH = exists_models[0]
    cfg.freeze()

    set_seed(1234)
    local_rank=args.local_rank


    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # output_dir = cfg.OUTPUT_DIR
    # print(output_dir)
    # if output_dir and not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # if rank==0:
    #     exists_models=glob(opj(output_dir,"*model.pth"))
    #     print(exists_models)
    #     exit()
    # print(cfg.MODEL.PRETRAIN_CHOICE)
    # exit()

    # if dist.get_rank() == 0:
    logger = setup_logger("reid_baseline", output_dir, if_train=True)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))

    # if cfg.MODEL.DEVICE == "cuda":
    #     os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    # cudnn.benchmark = True
    train(cfg,logger,local_rank)

if __name__ == '__main__':
    main()



    