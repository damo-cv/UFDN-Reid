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
from layers import NSSD_loss,NSSD_loss_2,NSSD_loss_3

from tensorboardX import SummaryWriter
import PIL, PIL.Image
import cv2
from tqdm import tqdm


def extract_feature(model, loader):
    features = torch.FloatTensor()
    all_pids = torch.LongTensor()
    all_camids = torch.LongTensor()
    all_tids = torch.LongTensor()

    for i, (inputs, pids, camids, tids, image_ids, _) in enumerate(loader):
        input_img = inputs.cuda()
        outputs = model(input_img, 120)
        ff = outputs.data.cpu()

        features = torch.cat((features, ff), 0)

        all_pids = torch.cat((all_pids, pids))
        all_camids = torch.cat((all_camids, camids))
        all_tids = torch.cat((all_tids, tids))
    return features, all_pids, all_camids, all_tids

def extract_feature_pcb(model, loader):
    features = torch.FloatTensor()
    features2 = torch.FloatTensor()
    features3 = torch.FloatTensor()
    features4 = torch.FloatTensor()
    all_pids = torch.LongTensor()
    all_camids = torch.LongTensor()
    all_tids = torch.LongTensor()

    for (inputs, pids, camids, tids, image_ids, _) in loader:
        input_img = inputs.cuda()
        feat, feat2, feat3 = model(input_img, 120)
        ff = feat.data.cpu()
        ff2 = feat2.data.cpu()
        ff3 = feat3.data.cpu()
        # ff4 = feat4.data.cpu()

        features = torch.cat((features, ff), 0)
        features2 = torch.cat((features2, ff2), 0)
        features3 = torch.cat((features3, ff3), 0)
        # features4 = torch.cat((features4, ff4), 0)
        all_pids = torch.cat((all_pids, pids))
        all_camids = torch.cat((all_camids, camids))
        all_tids = torch.cat((all_tids, tids))
    return features, features2, features3, all_pids, all_camids, all_tids

def reshape_transform(tensor, height=7, width=7):
    # print(tensor)
    # exit()

    # print(tensor.size())
    # min_value=torch.min(tensor)
    # tensor=tensor-min_value
    # print(tensor)
    # exit()
    try:
        result = tensor[:,1:,:].reshape(tensor.size(0),
                                height, width, tensor.size(2))
    except:
        result = tensor[:,:,:].reshape(tensor.size(0),
                                height, width, tensor.size(2))        

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def do_train_pd(
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
        nssd_weight_dict,
        writer=None
    ):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger.info("Start training")
    model.to(local_rank)
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs for training'.format(torch.cuda.device_count()))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    num_val = num_query + num_gallery

    if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        test=0
    else:
        test=1

    DF_mode=cfg.MODEL.TDeep
    scaler = amp.GradScaler()
    if cfg.MODEL.decouple_loss==2:
        nssd_loss=NSSD_loss(cfg.MODEL.num_stripes,cfg.MODEL.nssd_feature_dim,
            cluster_center_weight=cfg.MODEL.nssd_center_weight,
            margin=cfg.MODEL.nssd_center_margin,
            weight_dict=nssd_weight_dict)
    elif cfg.MODEL.decouple_loss==3:
        nssd_loss2=NSSD_loss_2(cfg.MODEL.num_stripes,cfg.MODEL.nssd_epoch,1024,
            cluster_center_weight=cfg.MODEL.nssd_center_weight,
            cluster_distance_weight=cfg.MODEL.nssd_distance_weight,
            weight_dict=nssd_weight_dict)
    elif cfg.MODEL.decouple_loss==5:
        nssd_loss=NSSD_loss_3(cfg.MODEL.num_stripes,cfg.MODEL.nssd_feature_dim,
            cluster_center_weight=cfg.MODEL.nssd_center_weight,
            margin=cfg.MODEL.nssd_center_margin,
            weight_dict=nssd_weight_dict)

    for epoch in range(start_epoch, epochs + 1):
        # print(epoch)
        # exit()
        if (epoch >(1-test) and epoch % eval_period == 1 or epoch == epochs):
            model.eval()
            with torch.no_grad():
                if cfg.MODEL.NAME=='PSwin' or cfg.MODEL.NAME=='PVit' or cfg.MODEL.NAME == 'Pres50':
                    feat, feat2,feat3, pids, camids, tids = extract_feature_pcb(model, val_loader)
                    g_features = feat[:num_val, :]
                    g_features2 = feat2[:num_val, :]
                    g_features3 = feat3[:num_val, :]
                    # print(g_features2.size())
                    # print(g_features3.size())
                    g_pids = pids[:num_val]
                    g_camids = camids[:num_val]
                    g_tids = tids[:num_val]

                    if cfg.MODEL.nssd_save_feature:
                        save_dict={}
                        save_dict["features"]=g_features2[:,2048:]
                        save_dict["pids"]=g_pids
                        np.save(os.path.join(cfg.OUTPUT_DIR,"test.npy"),save_dict)
                        print("successfully save the results")
                        exit()

                    metrics = R1_mAP(num_query, max_rank=100, feat_norm=cfg.TEST.FEAT_NORM, IF_mINP=True)
                    metrics.reset()
                    metrics.update((g_features, g_pids, g_camids, g_tids))
                    cmc, mAP, mINP = metrics.compute()

                    metrics2 = R1_mAP(num_query, max_rank=100, feat_norm=cfg.TEST.FEAT_NORM, IF_mINP=True)
                    metrics2.reset()
                    metrics2.update((g_features2, g_pids, g_camids, g_tids))
                    cmc_2, mAP_2, mINP_2 = metrics2.compute()

                    metrics3 = R1_mAP(num_query, max_rank=100, feat_norm=cfg.TEST.FEAT_NORM, IF_mINP=True)
                    metrics3.reset()
                    metrics3.update((g_features3, g_pids, g_camids, g_tids))
                    cmc_3, mAP_3, mINP_3 = metrics3.compute()

                    logger.info('Global Results')
                    logger.info("mAP: {:.1%}".format(mAP))
                    logger.info("mINP: {:.1%}".format(mINP))
                    for r in [1, 5, 10, 100, 1000]:
                        if r-1 >= len(cmc): continue
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                    logger.info('decouple Merged Results')
                    logger.info("mAP_2: {:.1%}".format(mAP_2))
                    logger.info("mINP_2: {:.1%}".format(mINP_2))
                    for r in [1, 5, 10, 100, 1000]:
                        if r-1 >= len(cmc_2): continue
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_2[r - 1]))

                    logger.info('All Merged Results')
                    logger.info("mAP_3: {:.1%}".format(mAP_3))
                    logger.info("mINP_3: {:.1%}".format(mINP_3))
                    for r in [1, 5, 10, 100, 1000]:
                        if r-1 >= len(cmc_3): continue
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_3[r - 1]))

                else:
                    features, pids, camids, tids = extract_feature(model, val_loader)
                    g_features = features[:num_val, :]
                    g_pids = pids[:num_val]
                    g_camids = camids[:num_val]
                    g_tids = tids[:num_val]

                    metrics = R1_mAP(num_query, max_rank=100, feat_norm=cfg.TEST.FEAT_NORM, IF_mINP=True,NUM_DECOUPLE=cfg.MODEL.NUM_DECOUPLE)
                    metrics.reset()
                    metrics.update((g_features, g_pids, g_camids, g_tids))
                    cmc, mAP, mINP = metrics.compute()
                    logger.info('Validation Results')
                    logger.info("mAP: {:.1%}".format(mAP))
                    logger.info("mINP: {:.1%}".format(mINP))
                    for r in [1, 5, 10, 100]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

        model.train()
        losses = AverageMeter()
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()

        loss_pid = AverageMeter()
        loss_color = AverageMeter()
        loss_type = AverageMeter()
        nssd_weight_dict=None
        if cfg.SOLVER.scheduler_mode=="Cos":
            scheduler.step(epoch)
        for i, (inputs, pids, camids, tids, distill_map, _) in enumerate(train_loader):
            if cfg.MODEL.PRETRAIN_CHOICE == 'evaluate':
                exit()
            num_steps = len(train_loader)
            n_iter = ( (epoch-1) * len(train_loader)) + i
            start_time = time.time()

            inputs = inputs.cuda()
            pids = pids.cuda()
            tids = tids.cuda()
            inputs = inputs.to(device) if torch.cuda.device_count() >= 1 else inputs
            pids = pids.to(device) if torch.cuda.device_count() >= 1 else pids
            tids = tids.to(device) if torch.cuda.device_count() >= 1 else tids
            if distill_map is not None:
                # distill_map = distill_map.cuda()
                distill_map=None

            
            if cfg.MODEL.NAME=='PSwin' or cfg.MODEL.NAME == 'Pres50':
                acc=0
                if cfg.MODEL.decouple_loss==0 or epoch<cfg.MODEL.nssd_epoch2:
                    scores, feats, image_feats = model(inputs,epoch)
                    loss_decouple=0
                elif cfg.MODEL.decouple_loss==3:
                    # with amp.autocast(enabled=True):
                    scores, feats, image_feats = model(inputs,epoch)
                    loss_decouple,nssd_weight_dict=nssd_loss2(image_feats[:,int(image_feats.size(1)/2):],i,len(train_loader),epoch)  
                elif cfg.MODEL.decouple_loss==2 or cfg.MODEL.decouple_loss==5:
                    scores, feats, image_feats = model(inputs,epoch)
                    loss_decouple,nssd_weight_dict=nssd_loss(image_feats[:,int(image_feats.size(1)/2):],i,len(train_loader),epoch)  
                else:
                    print("we only support triplet_loss2 or nssd_loss!")
                    exit()

                loss_reid=0
                for idx in range(len(feats)):
                    g_score = scores[idx]
                    g_feat = feats[idx]
                    cur_loss=loss_func(g_score, g_feat,pids, tids)
                    if cfg.MODEL.reid_loss_mode==1:
                        if idx==0:
                            loss_reid += cur_loss
                        else:
                            loss_reid += cur_loss / len(feats)
                    else:
                        loss_reid += cur_loss / len(feats)
                    acc += (g_score.max(1)[1] == pids).float().mean() / len(feats)
                    # exit()
                loss=loss_reid+loss_decouple
            else:
                loss_reid=0
                loss_decouple=0
                with amp.autocast(enabled=True):
                    score, feat= model(inputs)
                    g_distill_map = None
                    loss = loss_func(score, feat, pids, g_tids)
                acc = (score.max(1)[1] == pids).float().mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if dist.get_rank() == 0 and writer is not None:
                writer.add_scalar('loss', loss.item(), n_iter)
                writer.add_scalar('loss_reid', loss_reid.item(), n_iter)
                writer.add_scalar('loss_decouple', loss_decouple.item(), n_iter)
                writer.add_scalar('lr', scheduler.get_lr()[0], n_iter)
                writer.add_scalar('accuracy', acc.item(), n_iter)

            losses.update(loss.item())
            try:
                losses1.update(loss_reid.item())
            except:
                losses1.update(loss_reid)
            try:
                losses2.update(loss_decouple.item())
            except:
                losses2.update(loss_decouple)
            accs.update(acc.item())
            end_time = time.time()
            batch_time.update(end_time - start_time)

            if cfg.SOLVER.scheduler_mode=="Cos":
                if i % log_period == 0 and dist.get_rank() == 0:
                    logger.info('Epoch[{0}]\t'
                        'Iteration[{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss_reid {loss1.val:.4f} ({loss1.avg:.4f})\t'
                        'Loss_decouple {loss2.val:.4f} ({loss2.avg:.4f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {acc.val:.3f} ({acc.avg:.3f})\t'
                        'LR {lr:.7f}'.format(
                        epoch, i, len(train_loader), batch_time=batch_time, loss1=losses1, loss2=losses2, loss=losses, acc=accs, lr=scheduler.get_lr(epoch)[0]))
            else:
                if i % log_period == 0 and dist.get_rank() == 0:
                    logger.info('Epoch[{0}]\t'
                        'Iteration[{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss_reid {loss1.val:.4f} ({loss1.avg:.4f})\t'
                        'Loss_decouple {loss2.val:.4f} ({loss2.avg:.4f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {acc.val:.3f} ({acc.avg:.3f})\t'
                        'LR {lr:.7f}'.format(
                        epoch, i, len(train_loader), batch_time=batch_time, loss1=losses1, loss2=losses2, loss=losses, acc=accs, lr=scheduler.get_lr()[0]))            
        if not cfg.SOLVER.scheduler_mode=="Cos":
            scheduler.step()

        if dist.get_rank() == 0:
            # now=time.time()
            save_file = opj(output_dir, '{}_model.pth'.format(cfg.MODEL.NAME))
            logger.info('saving model {}'.format(save_file))
            save_state = {
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'last_epoch': epoch,
                'nssd_weight': nssd_weight_dict,
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
                'nssd_weight': nssd_weight_dict,
            }
            torch.save(save_state, save_file)
    exit()
