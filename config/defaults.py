from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# _C.MODEL.NAME = 'se_resnext50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'
_C.MODEL.IF_WITH_PCB = 'no'
_C.MODEL.IF_DOMAIN = 'no'
_C.MODEL.DOMAIN_TYPE = 'None'
_C.MODEL.IF_BFE = 'no'
_C.MODEL.IF_GZSZ = 'no'
_C.MODEL.IF_VIEWPOINT = 'no'
_C.MODEL.IF_MULTASKS = 'no'
_C.MODEL.IF_METRIC = 'no'
_C.MODEL.IF_ROI = 'no'
_C.MODEL.IF_METRIC_ALL = 'no'
_C.MODEL.IF_RPN = 'no'
_C.MODEL.IF_KEYPOINT = 'no'
_C.MODEL.IF_AUTO_KEYPOINT = 'no'
_C.MODEL.IF_DISTILL = 'no'
_C.MODEL.IF_GROUP = 'no'
_C.MODEL.IF_SVRN = 'no'
_C.MODEL.IF_GCN = 'no'
_C.MODEL.IF_TRANSFORMER = 'no'
_C.MODEL.IF_DISTILL_MUL = 'no'
_C.MODEL.IF_XBM = 'no'
_C.MODEL.IF_PD = 'no'
_C.MODEL.IF_OFFLINE_DISTILL = 'no'
_C.MODEL.IF_DECOUPLE = 'no'
_C.MODEL.IF_MUL = 'no'
# The loss type of metric loss
# options:'triplet','cluster','triplet_cluster','center','range_center','triplet_center','triplet_range_center'
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# For example, if loss type is cross entropy loss + triplet loss + center loss
# the setting should be: _C.MODEL.METRIC_LOSS_TYPE = 'triplet_center' and _C.MODEL.IF_WITH_CENTER = 'yes'

# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'

_C.MODEL.LOSS = ['focal', 'triplet']
_C.MODEL.LOSS_LOCAL = ['focal']
_C.MODEL.LOSS_WEIGHT = [1., 1.]
_C.MODEL.STATE_EPOCH = 1
_C.MODEL.GROUP_NUM = 4

_C.MODEL.Dstudent_model = ""
_C.MODEL.Dteacher_model = ""
_C.MODEL.SReduct = 1
_C.MODEL.TReduct = 1
_C.MODEL.TFreeze = False
_C.MODEL.num_stripes=1

_C.MODEL.LOSS_MULG = False
_C.MODEL.XMB_Memory = 768
_C.MODEL.XMB_EPOCH = 0
_C.MODEL.PMode = False
_C.MODEL.nssd_center_weight = [0.9,0.1]
_C.MODEL.nssd_center_margin=0.0
_C.MODEL.nssd_distance_weight = [0.9,0.1]
_C.MODEL.nssd_epoch = 60
_C.MODEL.nssd_epoch2 = 0
_C.MODEL.nssd_feature_dim=1024
_C.MODEL.nssd_concat=0
_C.MODEL.nssd_save_feature=False
_C.MODEL.nssd_use_local_feat=True
_C.MODEL.nssd_grad_cam=0
_C.MODEL.DThead_multi=0

_C.MODEL.TDeep = 0
_C.MODEL.Swin_model = 1
_C.MODEL.decouple_loss = 0
_C.MODEL.NUM_DECOUPLE = 1
_C.MODEL.block_only_depth = 2
_C.MODEL.freeze_base = 0

_C.MODEL.OFFLINE_PATH = ""


_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.STRIDE_SIZE = 16

_C.MODEL.mul_mode=0
_C.MODEL.reid_loss_mode=0
# # Transformer setting
# _C.MODEL.DROP_PATH = 0.1
# _C.MODEL.DROP_OUT = 0.0
# _C.MODEL.ATT_DROP_RATE = 0.0
# _C.MODEL.TRANSFORMER_TYPE = 'None'
# _C.MODEL.STRIDE_SIZE = [16, 16]

# JPM Parameter
_C.MODEL.JPM = False
_C.MODEL.SHIFT_NUM = 5
_C.MODEL.SHUFFLE_GROUP = 2
_C.MODEL.DEVIDE_LENGTH = 4
_C.MODEL.RE_ARRANGE = True

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('./data')
_C.DATASETS.filter = False
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16
_C.DATALOADER.NUM_CAMERAS = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 50
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 2
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3

_C.SOLVER.TRIPLET_WEIGHT = 1.

_C.SOLVER.CLS_WEIGHT = 1.

# _C.SOLVER.MARGIN_RANK = 1.3  ### R = ALPHA - MARGIN_RANK
# _C.SOLVER.ALPHA = 2.0
# _C.SOLVER.TVAL = 1.0


_C.SOLVER.MARGIN_RANK = 0.3  ### R = ALPHA - MARGIN_RANK
_C.SOLVER.ALPHA = 0.7
_C.SOLVER.TVAL = 10

_C.SOLVER.PAD_SIZE = 0
_C.SOLVER.PROPOSAL_NUM = 4
_C.SOLVER.TOP_N = 4

# _C.SOLVER.MARGIN_RANK = 0.4  ### R = ALPHA - MARGIN_RANK
# _C.SOLVER.ALPHA = 1.2
# _C.SOLVER.TVAL = 10

_C.SOLVER.METRIC_NUM = 15

_C.SOLVER.TRACK_MARGIN = 0.1
_C.SOLVER.TRACK_WEIGHT = 0.1
# Margin of cluster loss
_C.SOLVER.CLUSTER_MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
_C.SOLVER.PCB_PART_NUM = 4
_C.SOLVER.GLOBAL_NUM = 0
_C.SOLVER.HW_NUM = 2
# Settings of range loss
_C.SOLVER.RANGE_K = 2
_C.SOLVER.RANGE_MARGIN = 0.3
_C.SOLVER.RANGE_ALPHA = 0
_C.SOLVER.RANGE_BETA = 1
_C.SOLVER.RANGE_LOSS_WEIGHT = 1

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

# timm optimizer
# _C.SOLVER.TIMM = False
# _C.SOLVER.DECAY_EPOCHS= 30
# _C.SOLVER.DECAY_RATE=0.1
# _C.SOLVER.MIN_LR = 5e-6
# _C.SOLVER.WARMUP_LR = 5e-7
# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (30, 55)

# warm up factor
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_FREEZE_EPOCH = 0
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.scheduler_mode = "linear"

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 50
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 50

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'yes','no'
_C.TEST.RE_RANKING = 'no'
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""

_C.QCONFIG = ""