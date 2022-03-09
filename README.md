# LSHFM.detection

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/in-defense-of-feature-mimicking-for-knowledge/knowledge-distillation-on-imagenet)](https://paperswithcode.com/sota/knowledge-distillation-on-imagenet?p=in-defense-of-feature-mimicking-for-knowledge)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/in-defense-of-feature-mimicking-for-knowledge/knowledge-distillation-on-coco)](https://paperswithcode.com/sota/knowledge-distillation-on-coco?p=in-defense-of-feature-mimicking-for-knowledge)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/in-defense-of-feature-mimicking-for-knowledge/knowledge-distillation-on-pascal-voc)](https://paperswithcode.com/sota/knowledge-distillation-on-pascal-voc?p=in-defense-of-feature-mimicking-for-knowledge)

This is the PyTorch source code for [Distilling Knowledge by Mimicking Features](https://arxiv.org/abs/2011.01424). We provide all codes for three tasks.

* single-label image classification: [LSHFM.singleclassification](https://github.com/DoctorKey/LSHFM.singleclassification)
* multi-label image classification: [LSHFM.multiclassification](https://github.com/DoctorKey/LSHFM.multiclassification)
* object detection: [LSHFM.detection](https://github.com/DoctorKey/LSHFM.detection)

## dependence

* python
* pytorch 1.7.1
* torchvision 0.8.2

## Prepare the dataset

Please prepare the COCO and VOC datasets by youself. Then you need to fix the `get_data_path` function in `src/dataset/coco_utils.py` and `src/dataset/voc_utils.py`. 

## Run

You can run the experiments by
```
PORT=4444 bash experiments/[script name].sh 0,1,2,3 
```

the training set contains VOC2007 trainval and VOC2012 trainval, while the testing set is VOC2007 test.

We train all models by 24 epochs while the learning rate decays at the 18th and 22th epoch.

### Faster R-CNN

Before you run the KD experiments, please make sure the teacher model weight have been saved in `pretrained`. You can first run `ResNet101 baseline` and `VGG16 baseline` to train the teacher model, and then move the model to `pretrained` and edit `--teacher-ckpt` in the training shell scripts. You can also download [voc0712_fasterrcnn_r101_83.6](https://box.nju.edu.cn/f/395da9c3b49644f1ad22/) and [voc0712_fasterrcnn_vgg16fpn_79.0](https://box.nju.edu.cn/f/c46346d07fd1426c877b/) directly, and move them to `pretrained`.

* ResNet101 baseline: [voc0712_fasterrcnn_r101_baseline.sh](experiments/voc0712_fasterrcnn_r101_baseline.sh)
* ResNet50 baseline: [voc0712_fasterrcnn_r50_baseline.sh](experiments/voc0712_fasterrcnn_r50_baseline.sh)
* ResNet50@ResNet101 L2: [voc0712_fasterrcnn_r50_r101_l2.sh](experiments/voc0712_fasterrcnn_r50_r101_l2.sh)
* ResNet50@ResNet101 LSH: [voc0712_fasterrcnn_r50_r101_lsh.sh](experiments/voc0712_fasterrcnn_r50_r101_lsh.sh)
* ResNet50@ResNet101 LSHL2: [voc0712_fasterrcnn_r50_r101_lshl2.sh](experiments/voc0712_fasterrcnn_r50_r101_lshl2.sh)

* VGG16 baseline: [voc0712_fasterrcnn_vgg11fpn_baseline.sh](experiments/voc0712_fasterrcnn_vgg11fpn_baseline.sh)
* VGG11 baseline: [voc0712_fasterrcnn_vgg16fpn_baseline.sh](experiments/voc0712_fasterrcnn_vgg16fpn_baseline.sh)
* VGG11@VGG16 L2: [voc0712_fasterrcnn_vgg11fpn_vgg16fpn_l2.sh](experiments/voc0712_fasterrcnn_vgg11fpn_vgg16fpn_l2.sh)
* VGG11@VGG16 LSH: [voc0712_fasterrcnn_vgg11fpn_vgg16fpn_lsh.sh](experiments/voc0712_fasterrcnn_vgg11fpn_vgg16fpn_lsh.sh)
* VGG11@VGG16 LSHL2: [voc0712_fasterrcnn_vgg11fpn_vgg16fpn_lshl2.sh](experiments/voc0712_fasterrcnn_vgg11fpn_vgg16fpn_lshl2.sh)

|       	| ResNet50@ResNet101 | VGG11@VGG16 |
| :---: 	| :----: 			 | :----: 	|
| Teacher 	|   83.6    		|   79.0    |
| Student 	|   82.0     		|   75.1    |
| L2 		|   83.0			|   76.8    |
| LSH 		|   82.6   			|   76.7    |
| LSHL2 	|   83.0    		|   77.2    | 

### RetinaNet

As mentioned in Faster R-CNN, please make sure there are teacher models in `pretrained`. You can download the teacher models in [voc0712_retinanet_r101_83.0.ckpt](https://box.nju.edu.cn/f/49b435b37f3e4894bfc3/) and [voc0712_retinanet_vgg16fpn_76.6.ckpt](https://box.nju.edu.cn/f/2eb8830ecad8493cb801/).

* ResNet101 baseline: [voc0712_retinanet_r101_baseline.sh](experiments/voc0712_retinanet_r101_baseline.sh)
* ResNet50 baseline: [voc0712_retinanet_r50_baseline.sh](experiments/voc0712_retinanet_r50_baseline.sh)
* ResNet50@ResNet101 L2: [voc0712_retinanet_r50_r101_l2.sh](experiments/voc0712_retinanet_r50_r101_l2.sh)
* ResNet50@ResNet101 LSHL2: [voc0712_retinanet_r50_r101_lshl2.sh](experiments/voc0712_retinanet_r50_r101_lshl2.sh)

* VGG16 baseline: [voc0712_retinanet_vgg11fpn_baseline.sh](experiments/voc0712_retinanet_vgg11fpn_baseline.sh)
* VGG11 baseline: [voc0712_retinanet_vgg16fpn_baseline.sh](experiments/voc0712_retinanet_vgg16fpn_baseline.sh)
* VGG11@VGG16 L2: [voc0712_retinanet_vgg11fpn_vgg16fpn_l2.sh](experiments/voc0712_retinanet_vgg11fpn_vgg16fpn_l2.sh)
* VGG11@VGG16 LSHL2: [voc0712_retinanet_vgg11fpn_vgg16fpn_lshl2.sh](experiments/voc0712_retinanet_vgg11fpn_vgg16fpn_lshl2.sh)

|       	| ResNet50@ResNet101 | VGG11@VGG16 |
| :---: 	| :----: 			 | :----: 	|
| Teacher 	|   83.0    		|   76.6    |
| Student 	|   82.5     		|   73.2    |
| L2 		|   82.6			|   74.8    |
| LSHL2 	|   83.0    		|   75.2    | 

We find that it is easy to get NaN loss when training by LSH KD. 


### visualize


visualize the ground truth label

```
python src/visual.py --dataset voc07 --idx 1 --gt
```

visualize the model prediction
```
python src/visual.py --dataset voc07 --idx 2 --model fasterrcnn_resnet50_fpn --checkpoint results/voc0712/fasterrcnn_resnet50_fpn/2020-12-11_20\:14\:09/model_13.pth
```

## Citing this repository

If you find this code useful in your research, please consider citing us:

```
@article{LSHFM,
  title={Distilling knowledge by mimicking features},
  author={Wang, Guo-Hua and Ge, Yifan and Wu, Jianxin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
}
```

## Acknowledgement

This project is based on [https://github.com/pytorch/vision/tree/master/references/detection](https://github.com/pytorch/vision/tree/master/references/detection). This project aims at object detection, so I remove the code about segmentation and keypoint detection.
