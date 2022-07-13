import argparse
import os
from os.path import join as opj
import sys
import time
import numpy as np
import collections

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.cuda import amp

sys.path.append('..')
from config import cfg
from data import make_data_loader
from layers import make_loss
from solver import make_optimizer, make_optimizer_with_center, make_optimizer_two_stream, WarmupMultiStepLR
from .AverageMeter import AverageMeter

from utils.logger import setup_logger
from utils.reid_metric import R1_mAP, Track_R1_mAP
# from layers import CrossEntropySoftLabel

# import spring.linklink as link
# from dist.distributed_utils import dist_init, reduce_gradients, DistModule, AllGather, AllReduce

from tensorboardX import SummaryWriter


def extract_feature(model, loader, device):
    features = torch.FloatTensor()
    all_pids = torch.LongTensor()
    all_camids = torch.LongTensor()
    all_tids = torch.LongTensor()

    for i, (inputs, pids, camids, tids, _, _) in enumerate(loader):
        # print(i,len(loader))
        # input_img = inputs.cuda()
        input_img = inputs.to(device) if torch.cuda.device_count() >= 1 else inputs
        outputs = model(input_img)
        ff = outputs.data.cpu()

        features = torch.cat((features, ff), 0)

        all_pids = torch.cat((all_pids, pids))
        all_camids = torch.cat((all_camids, camids))
        all_tids = torch.cat((all_tids, tids))
    return features, all_pids, all_camids, all_tids

# def extract_feature_pcb(model, loader):
#     features = torch.FloatTensor()
#     features2 = torch.FloatTensor()
#     features3 = torch.FloatTensor()
#     features4 = torch.FloatTensor()
#     all_pids = torch.LongTensor()
#     all_camids = torch.LongTensor()
#     all_tids = torch.LongTensor()

#     for (inputs, pids, camids, tids, _, _) in loader:
#         input_img = inputs.cuda()
#         feat, feat2 = model(input_img, pids)
#         ff = feat.data.cpu()
#         ff2 = feat2.data.cpu()
#         # ff3 = feat3.data.cpu()
#         # ff4 = feat4.data.cpu()

#         features = torch.cat((features, ff), 0)
#         features2 = torch.cat((features2, ff2), 0)
#         # features3 = torch.cat((features3, ff3), 0)
#         # features4 = torch.cat((features4, ff4), 0)
#         all_pids = torch.cat((all_pids, pids))
#         all_camids = torch.cat((all_camids, camids))
#         all_tids = torch.cat((all_tids, tids))
#     return features, features2, all_pids, all_camids, all_tids

def do_train(
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
        writer=None
    ):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    if cfg.MODEL.PRETRAIN_CHOICE == 'evaluate':
        test=1
    else:
        test=0

    logger.info("Start training")
    model.to(local_rank)
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs for training'.format(torch.cuda.device_count()))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    # print("after initiallizing")
    num_val = num_query + num_gallery
    # model = nn.DataParallel(model)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    # model.to(device)
    # model = DistModule(model, sync=True)
    scaler = amp.GradScaler()
    for epoch in range(start_epoch, epochs + 1):
        if (epoch >1 and epoch % eval_period == 1 or epoch == epochs):
        # if (epoch >(1-test) and epoch % eval_period == 1 or epoch == epochs):
            model.eval()
            g_features, g_pids, g_camids, g_tids = extract_feature(model, val_loader,device)
            print("generating feature")
            
            g_features = g_features[:num_val, :]
            g_pids = g_pids[:num_val]
            g_camids = g_camids[:num_val]
            g_tids = g_tids[:num_val]
            print("computing mAP")

            metrics = R1_mAP(num_query, max_rank=100, feat_norm=cfg.TEST.FEAT_NORM, IF_mINP=True,NUM_DECOUPLE=cfg.MODEL.NUM_DECOUPLE)
            metrics.reset()
            metrics.update((g_features, g_pids, g_camids, g_tids))
            cmc, mAP, mINP = metrics.compute()
            if dist.get_rank()==0:
                logger.info('Validation Results')
                logger.info("mAP: {:.1%}".format(mAP))
                logger.info("mINP: {:.1%}".format(mINP))
                for r in [1, 5, 10, 100]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        # scheduler.step()
        model.train()
        scheduler.step(epoch)
        losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()

        loss_pid = AverageMeter()
        loss_color = AverageMeter()
        loss_type = AverageMeter()
        for i, (inputs, pids, camids, tids, distill_map, _) in enumerate(train_loader):
            # exit()
            # optimizer.zero_grad()
            if cfg.MODEL.PRETRAIN_CHOICE == 'evaluate':
                exit()
            num_steps = len(train_loader)
            n_iter = ( (epoch-1) * len(train_loader)) + i
            start_time = time.time()

            inputs = inputs.to(device) if torch.cuda.device_count() >= 1 else inputs
            pids = pids.to(device) if torch.cuda.device_count() >= 1 else pids
            tids = tids.to(device) if torch.cuda.device_count() >= 1 else tids
            with amp.autocast(enabled=True):
                score, feat= model(inputs,epoch)
                loss = loss_func(score, feat, pids, tids)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            acc = (score.max(1)[1] == pids).float().mean()

            if dist.get_rank() == 0 and writer is not None:
                writer.add_scalar('loss', loss.item(), n_iter)
                writer.add_scalar('lr', scheduler.get_lr()[0], n_iter)
                writer.add_scalar('accuracy', acc.item(), n_iter)

            losses.update(loss.item())

            accs.update(acc.item())
            end_time = time.time()
            batch_time.update(end_time - start_time)

            torch.cuda.synchronize()

            if i % log_period == 0 and dist.get_rank() == 0:
                logger.info('Epoch[{0}]\t'
                    'Iteration[{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {acc.val:.3f} ({acc.avg:.3f})\t'
                    'LR {lr:.7f}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, loss=losses, acc=accs, lr=scheduler.get_lr(epoch)[0]))
        # scheduler.step()

        if dist.get_rank()==0:
            # now=time.time()
            save_file = opj(output_dir, '{}_model.pth'.format(cfg.MODEL.NAME))
            logger.info('saving model {}'.format(save_file))
            save_state = {
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'last_epoch': epoch,
            }
            torch.save(save_state, save_file)
            # print("save successfull",time.time()-now)
            # exit()
        if epoch % checkpoint_period == 0 and dist.get_rank() == 0:
            save_file = opj(output_dir, '{}_model_{}.pth'.format(cfg.MODEL.NAME, epoch))
            logger.info('saving model {}'.format(save_file))
            save_state = {
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'last_epoch': epoch,
            }
            torch.save(save_state, save_file)