import torch
from .ranger import * 

def make_optimizer(cfg, model,paras):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            # print(key)
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if paras:
        params+=[{"params": [paras[0],paras[1]]}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        # filter(lambda p: p.requires_grad, params)
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, eps=1e-8, betas=(0.9, 0.999))
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Ranger':
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr}]
        optimizer = Ranger(params)
        # optimizer = RangerVA(params)
        # optimizer = RangerQH(params)
    return optimizer

def make_optimizer_two_stream(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if 'layer4_p' in key or 'reduction' in key or 'classifier' in key or 'bnneck' in key: lr = 2 * lr
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    return optimizer

def make_optimizer_domain(cfg, model):
    params = []
    params_domain = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if (cfg.MODEL.DOMAIN_TYPE == 'MULFC' or cfg.MODEL.DOMAIN_TYPE == 'DAN') and 'base' not in key: lr = 2 * lr
        if 'domain' not in key:
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        else:
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            params_domain += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
        if len(params_domain) != 0: 
            optimizer_domain = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_domain, momentum=cfg.SOLVER.MOMENTUM)
        else: optimizer_domain = None
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
        if len(params_domain) != 0: 
            optimizer_domain = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_domain)
        else: 
            optimizer_domain = None
    return [optimizer, optimizer_domain]

def make_optimizer_with_center(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer, optimizer_center