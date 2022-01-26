#!/usr/bin/env bash

# run by PORT=4444 bash fasterrcnn_r50_r101_lsh.sh 4,5,6,7
GPUS=$1
PORT=${PORT}

CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --master_port $PORT --nproc_per_node=4 --use_env \
	src/train.py --dataset voc0712 --model fasterrcnn_vgg11fpn_with_vgg16fpn \
	--epochs 24 --lr-steps 18 22 --lr 0.01 --lr-warmup 1000 \
	--min-size 480 512 544 576 608 640 672 704 736 768 800 \
	--backbone-pretrained https://download.pytorch.org/models/vgg11-bbd30ac9.pth \
	--teacher-ckpt pretrained/voc0712_fasterrcnn_vgg16fpn_79.0.ckpt \
	--distill lshl2 --beta 7 --bias 0