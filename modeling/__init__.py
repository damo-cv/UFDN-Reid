from .baseline import Baseline
from .baseline_pd import Baseline as Baseline_pd


def build_model(cfg, num_classes):
    if cfg.MODEL.IF_PD == 'yes':
        model = Baseline_pd(cfg, num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    else:
        model = Baseline(cfg, num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    return model