#!/usr/bin/env bash

# run by:
# PORT=4444 bash experiments/faster_rcnn_resnet50_baseline.sh 0,1,2,3 
GPUS=$1
PORT=${PORT}

CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --master_port $PORT --nproc_per_node=4 --use_env \
	src/train.py --dataset voc07 --model retinanet_vgg16 --epochs 18 --lr-steps 12 16 --lr 0.005 --lr-warmup 1000 \
    --min-size 800 \
	--backbone-pretrained https://download.pytorch.org/models/vgg16-397923af.pth