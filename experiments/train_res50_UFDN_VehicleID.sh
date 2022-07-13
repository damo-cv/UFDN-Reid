CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12344 tools/dist_train.py \
--config_file='configs/softmax_triplet_pd.yaml' MODEL.DEVICE_ID "('0')" MODEL.TDeep "(2)" MODEL.num_stripes "(4)" \
MODEL.PRETRAIN_PATH "('/home/qianwen.qian/model/model_pretrain/resnet50-19c8e357.pth')" MODEL.NAME "('Pres50')" \
INPUT.SIZE_TRAIN "([224, 224])" INPUT.SIZE_TEST "([224, 224])" DATASETS.NAMES "('VehicleID')" MODEL.nssd_center_weight "([0.6, 0.4])" MODEL.nssd_feature_dim "(2048)" \
OUTPUT_DIR "('/home/qianwen.qian/model/model_zoo/UFDN/baseline_VID_Pres50_decouple_001')" \
SOLVER.MAX_EPOCHS "(120)" MODEL.LOSS_LOCAL "(['focal', 'triplet'])" MODEL.decouple_loss "(2)" \
SOLVER.IMS_PER_BATCH "(128)" TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('yes')" DATALOADER.NUM_INSTANCE "4" \
SOLVER.STEPS "([40, 70])" SOLVER.WARMUP_ITERS "(10)" SOLVER.BASE_LR "(3e-4)" SOLVER.OPTIMIZER_NAME "('Adam')" SOLVER.scheduler_mode "('Cos')"