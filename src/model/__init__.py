from model.faster_rcnn_vgg import fasterrcnn_vgg11, fasterrcnn_vgg16
from model.faster_rcnn_vgg import fasterrcnn_vgg11_fpn, fasterrcnn_vgg16_fpn
from model.faster_rcnn_resnet import fasterrcnn_resnet50_fpn, fasterrcnn_resnet101_fpn
from model.faster_rcnn_resnet import fasterrcnn_resnet50_c4, fasterrcnn_resnet101_c4
from model.faster_rcnn_KD import fasterrcnn_r50_with_r101, fasterrcnn_vgg11_with_vgg16, fasterrcnn_vgg11fpn_with_vgg16fpn

from model.retinanet_vgg import retinanet_vgg11, retinanet_vgg16
from model.retinanet_vgg import retinanet_vgg11_fpn, retinanet_vgg16_fpn
from model.retinanet_resnet import retinanet_resnet50_fpn, retinanet_resnet101_fpn
from model.retinanet_KD import retinanet_r50_with_r101, retinanet_vgg11fpn_with_vgg16fpn

model_dict = {
    'fasterrcnn_vgg16': fasterrcnn_vgg16,
    'fasterrcnn_vgg11': fasterrcnn_vgg11,
    'fasterrcnn_vgg11_with_vgg16': fasterrcnn_vgg11_with_vgg16,
    'fasterrcnn_vgg11_fpn': fasterrcnn_vgg11_fpn,
    'fasterrcnn_vgg16_fpn': fasterrcnn_vgg16_fpn,
    'fasterrcnn_vgg11fpn_with_vgg16fpn': fasterrcnn_vgg11fpn_with_vgg16fpn,

    'fasterrcnn_resnet50_c4': fasterrcnn_resnet50_c4,
    'fasterrcnn_resnet101_c4': fasterrcnn_resnet101_c4,
    'fasterrcnn_resnet50_fpn':fasterrcnn_resnet50_fpn,
    'fasterrcnn_resnet101_fpn':fasterrcnn_resnet101_fpn,
    'fasterrcnn_r50_with_r101': fasterrcnn_r50_with_r101,
    
    'retinanet_resnet50_fpn': retinanet_resnet50_fpn,
    'retinanet_resnet101_fpn': retinanet_resnet101_fpn,
    'retinanet_r50_with_r101': retinanet_r50_with_r101,
    
    'retinanet_vgg11': retinanet_vgg11,
    'retinanet_vgg11_fpn': retinanet_vgg11_fpn,
    'retinanet_vgg16': retinanet_vgg16,
    'retinanet_vgg16_fpn': retinanet_vgg16_fpn,
    'retinanet_vgg11fpn_with_vgg16fpn': retinanet_vgg11fpn_with_vgg16fpn,
}
