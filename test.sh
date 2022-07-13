# CUDA_VISIBLE_DEVICES=0,1 
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 tools/dist_train.py \
--config_file='configs/softmax_triplet.yaml' MODEL.DEVICE_ID "('0')" \
MODEL.PRETRAIN_PATH "('/home/qianwen.qian/model/model_pretrain/resnet50-19c8e357.pth')" MODEL.NAME "('resnet50')" \
INPUT.SIZE_TRAIN "([256, 256])" INPUT.SIZE_TEST "([256, 256])" DATASETS.NAMES "('VeRi')" \
OUTPUT_DIR "('/home/qianwen.qian/model/model_zoo/demo_reduct_test')" \
SOLVER.IMS_PER_BATCH "(128)" TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('yes')" DATALOADER.NUM_INSTANCE "8" SOLVER.BASE_LR "(1e-4)" 