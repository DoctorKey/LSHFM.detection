#!/usr/bin/env bash

# run by:
# PORT=4444 bash experiments/faster_rcnn_resnet50_baseline.sh 0,1,2,3 
GPUS=$1
PORT=${PORT}

CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --master_port $PORT --nproc_per_node=4 --use_env \
	src/train.py --dataset voc0712 --model retinanet_vgg11fpn_with_vgg16fpn \
	--epochs 24 --lr-steps 18 22 --lr 0.005 --lr-warmup 1000 \
    --min-size 480 512 544 576 608 640 672 704 736 768 800 \
	--backbone-pretrained https://download.pytorch.org/models/vgg11-bbd30ac9.pth \
	--teacher-ckpt pretrained/voc0712_retinanet_vgg16fpn_76.6.ckpt \
    --distill lsh --beta 6 --bias 0 --hash-num 4096