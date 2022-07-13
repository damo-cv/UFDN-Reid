# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 tools/dist_train.py \
# --config_file='configs/softmax_triplet_pd.yaml' MODEL.DEVICE_ID "('0')" MODEL.TDeep "(2)" MODEL.Swin_model "(0)" MODEL.num_stripes "(4)" \
# MODEL.PRETRAIN_PATH "('/home/qianwen.qian/model/model_pretrain/swin_tiny_patch4_window7_224.pth')" MODEL.NAME "('PSwin')" \
# INPUT.SIZE_TRAIN "([224, 224])" INPUT.SIZE_TEST "([224, 224])" DATASETS.NAMES "('VehicleID')" MODEL.nssd_center_weight "([0.9, 0.1])" MODEL.nssd_feature_dim "(2048)" \
# OUTPUT_DIR "('/home/qianwen.qian/model/model_zoo/UFDN/baseline_VID_Pswin_decouple_001')" MODEL.nssd_epoch2 "(10)" \
# SOLVER.MAX_EPOCHS "(120)" MODEL.LOSS_LOCAL "(['focal', 'triplet'])" MODEL.decouple_loss "(5)" MODEL.nssd_center_margin "(0.1)" \
# SOLVER.IMS_PER_BATCH "(64)" TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('yes')" DATALOADER.NUM_INSTANCE "4" \
# SOLVER.STEPS "([50, 80])" SOLVER.WARMUP_ITERS "(20)" SOLVER.BASE_LR "(3e-4)" SOLVER.OPTIMIZER_NAME "('AdamW')" SOLVER.scheduler_mode "('Cos')"


CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12346 tools/dist_train.py \
--config_file='configs/softmax_triplet_pd.yaml' MODEL.DEVICE_ID "('0')" MODEL.TDeep "(2)" MODEL.Swin_model "(0)" MODEL.num_stripes "(4)" \
MODEL.PRETRAIN_PATH "('/home/qianwen.qian/model/model_pretrain/swin_tiny_patch4_window7_224.pth')" MODEL.NAME "('PSwin')" \
INPUT.SIZE_TRAIN "([224, 224])" INPUT.SIZE_TEST "([224, 224])" DATASETS.NAMES "('VehicleID')" MODEL.nssd_center_weight "([0.9, 0.1])" MODEL.nssd_feature_dim "(2048)" \
OUTPUT_DIR "('/home/qianwen.qian/model/model_zoo/UFDN/baseline_VID_Pswin_decouple_002')" MODEL.nssd_epoch2 "(10)" \
SOLVER.MAX_EPOCHS "(120)" MODEL.LOSS_LOCAL "(['focal', 'triplet'])" MODEL.decouple_loss "(5)" MODEL.nssd_center_margin "(0.1)" \
SOLVER.IMS_PER_BATCH "(64)" TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('yes')" DATALOADER.NUM_INSTANCE "4" \
SOLVER.STEPS "([50, 80])" SOLVER.WARMUP_ITERS "(20)" SOLVER.BASE_LR "(3e-4)" SOLVER.OPTIMIZER_NAME "('AdamW')" SOLVER.scheduler_mode "('Cos')"