#!/usr/bin/env bash

# run by PORT=4444 bash faster_rcnn_resnet50_baseline.sh 0,1,2,3 
GPUS=$1
PORT=${PORT}

CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --master_port $PORT --nproc_per_node=4 --use_env \
	src/train.py --dataset voc0712 --model fasterrcnn_vgg11 --epochs 24 --lr-steps 18 22 --lr 0.01 --lr-warmup 1000 \
	--min-size 480 512 544 576 608 640 672 704 736 768 800 \
	--backbone-pretrained https://download.pytorch.org/models/vgg11-bbd30ac9.pth