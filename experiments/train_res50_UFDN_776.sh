CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 tools/dist_train.py \
--config_file='configs/softmax_triplet_pd.yaml' MODEL.DEVICE_ID "('0')" MODEL.TDeep "(2)" MODEL.num_stripes "(4)" \
MODEL.PRETRAIN_PATH "('/home/qianwen.qian/model/model_pretrain/resnet50-19c8e357.pth')" MODEL.NAME "('Pres50')" \
INPUT.SIZE_TRAIN "([224, 224])" INPUT.SIZE_TEST "([224, 224])" DATASETS.NAMES "('VeRi')" MODEL.nssd_center_weight "([0.9, 0.1])" MODEL.nssd_feature_dim "(2048)" \
OUTPUT_DIR "('/home/qianwen.qian/model/model_zoo/UFDN/baseline_veri_Pres50_decouple_004')" \
SOLVER.MAX_EPOCHS "(120)" MODEL.LOSS_LOCAL "(['focal', 'triplet'])" MODEL.decouple_loss "(2)" \
SOLVER.IMS_PER_BATCH "(64)" TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('yes')" DATALOADER.NUM_INSTANCE "16" \
SOLVER.STEPS "([40, 70])" SOLVER.WARMUP_ITERS "(10)" SOLVER.BASE_LR "(3e-4)" SOLVER.OPTIMIZER_NAME "('Adam')" SOLVER.scheduler_mode "('linear')"
# 81.5%