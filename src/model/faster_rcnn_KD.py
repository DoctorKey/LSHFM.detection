"""
Implements the knowledge distillation for faster rcnn
"""

from collections import OrderedDict
from typing import Union
import torch
from torch import nn
import warnings
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor
import torchvision

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
#from torchvision.models.detection.faster_rcnn import FasterRCNN
from model.vgg_fpn import vgg_fpn_backbone
from model.faster_rcnn import FasterRCNN
from model.KD import create_KD_loss
from model.utils import load_backbone_pretrained


class FasterRCNNKD(nn.Module):
    """
    Main class for faster rcnn KD
    """
    def __init__(self, teacher, student, opt):
        super(FasterRCNNKD, self).__init__()
        # the teacher will not update
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        self.t_transform = teacher.transform
        self.t_backbone = teacher.backbone
        self.t_rpn = teacher.rpn
        self.t_roi_heads = teacher.roi_heads
        
        self.s_transform = student.transform
        self.s_backbone = student.backbone
        self.s_rpn = student.rpn
        self.s_roi_heads = student.roi_heads

        if opt['hash_num'] is None:
            opt['hash_num'] = opt['feature_dim'] * 4
        if opt['std'] is None:
            opt['std'] = self.t_roi_heads.box_predictor.cls_score.weight.data.std()
        self.KD_loss = create_KD_loss(opt)


    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels`.
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # use student's transform to resize images
        images, targets = self.s_transform(images, targets)
  
        # only compute student
        s_feature = self.s_backbone(images.tensors)
        if isinstance(s_feature, torch.Tensor):
            s_feature = OrderedDict([('0', s_feature)])
        s_proposals, s_proposal_losses = self.s_rpn(images, s_feature, targets)
        s_detections, s_detector_losses, s_proposals, s_box_raw_features, s_box_features = self.s_roi_heads(
            s_feature, s_proposals, images.image_sizes, targets, KD='student')
        s_detections = self.s_transform.postprocess(s_detections, images.image_sizes, original_image_sizes)

        # eval the student
        if not self.training:
            return s_detections

        with torch.no_grad():
            t_feature = self.t_backbone(images.tensors)
            if isinstance(t_feature, torch.Tensor):
                t_feature = OrderedDict([('0', t_feature)])
            t_proposals, t_proposal_losses = self.t_rpn(images, t_feature, targets)
            t_proposals = s_proposals
            t_detections, t_detector_losses, t_proposals, t_box_raw_features, t_box_features = self.t_roi_heads(
                t_feature, t_proposals, images.image_sizes, targets, KD='teacher')
            t_detections = self.t_transform.postprocess(t_detections, images.image_sizes, original_image_sizes)

        kd_loss = self.KD_loss(s_box_features, t_box_features)
        kd_losses = {'kd_loss': kd_loss}

        losses = {}
        losses.update(s_detector_losses)
        losses.update(s_proposal_losses)
        losses.update(kd_losses)

        return losses


def load_checkpoint(model, ckpt=None, teacher=True):
    if teacher and not ckpt:
        print('WARNING: none teacher pretrained weight!!!')
        return
    if ckpt is None:
        return
    print("=> loading pretrained from checkpoint {}".format(ckpt))
    state_dict = torch.load(ckpt, map_location='cpu')['model']
    ret = model.load_state_dict(state_dict)
    head = 'teacher' if teacher else 'student'
    print('{} load result:{}'.format(head, ret))


def fasterrcnn_r50_with_r101(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, 
                            student_ckpt=None, teacher_ckpt=None, KD_opt=None, **kwargs):
    assert KD_opt is not None, 'need opt to define KD loss'
    if trainable_backbone_layers is None:
        trainable_backbone_layers = 3
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0

    if student_ckpt is None or not backbone_pretrained:
        load_imagenet = True
    else:
        load_imagenet = False
    s_backbone = resnet_fpn_backbone('resnet50', load_imagenet, trainable_layers=trainable_backbone_layers)
    if backbone_pretrained:
        load_backbone_pretrained(s_backbone, backbone_pretrained, 'resnet_fpn')
    # In our KD framework, images will be transform by student
    if min_size is None:
        min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    rpn_pre_nms_top_n_train=4000
    rpn_pre_nms_top_n_test=2000
    r50 = FasterRCNN(s_backbone, num_classes, min_size=min_size, 
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test)
    if student_ckpt:
        load_checkpoint(r50, ckpt=student_ckpt, teacher=False)

    # teacher backbone does not need to load
    t_backbone = resnet_fpn_backbone('resnet101', False, trainable_layers=0)
    r101 = FasterRCNN(t_backbone, num_classes, min_size=min_size, 
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test)
    load_checkpoint(r101, ckpt=teacher_ckpt, teacher=True)

    #KD_opt['feature_dim'] = 256 * 7 ** 2
    KD_opt['feature_dim'] = 1024
    if KD_opt['std'] is None:
        KD_opt['std'] = r101.roi_heads.box_predictor.cls_score.weight.data.std()
    if KD_opt['hash_num'] is None:
        KD_opt['hash_num'] = KD_opt['feature_dim'] * 4
    model = FasterRCNNKD(teacher=r101, student=r50, opt=KD_opt)
    return model


def fasterrcnn_vgg11_with_vgg16(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, 
                            student_ckpt=None, teacher_ckpt=None, KD_opt=None, **kwargs):
    assert KD_opt is not None, 'need opt to define KD loss'
    if student_ckpt is None or not backbone_pretrained:
        load_imagenet = True
    else:
        load_imagenet = False
    backbone = torchvision.models.vgg11(pretrained=load_imagenet)
    if backbone_pretrained:
        load_backbone_pretrained(backbone, backbone_pretrained, 'vgg')
    features = list(backbone.features)[:20]

    if trainable_backbone_layers is None:
        trainable_backbone_layers = 14
    assert trainable_backbone_layers <= 20 and trainable_backbone_layers >= 0
    # freeze top4 conv
    for layer in features[:20 - trainable_backbone_layers]:
        for p in layer.parameters():
            p.requires_grad = False

    backbone = nn.Sequential(*features)
    backbone.out_channels = 512
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))


    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
                                                    output_size=7,
                                                    sampling_ratio=2)
    if min_size is None:
        min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    rpn_pre_nms_top_n_train=4000
    rpn_pre_nms_top_n_test=2000
    vgg11 = FasterRCNN(backbone,
                       num_classes=num_classes, min_size=min_size,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler, 
                       rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train, 
                       rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test)
    if student_ckpt:
        load_checkpoint(vgg11, ckpt=student_ckpt, teacher=False)


    backbone = torchvision.models.vgg16(pretrained=False)
    # the 30th layer of features is relu of conv5_3
    features = list(backbone.features)[:30]

    backbone = nn.Sequential(*features)

    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
                                                    output_size=7,
                                                    sampling_ratio=2)
    vgg16 = FasterRCNN(backbone,
                       num_classes=num_classes, min_size=min_size,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler, 
                       rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train, 
                       rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test)
    load_checkpoint(vgg16, ckpt=teacher_ckpt, teacher=True)
    # freeze all
    for p in vgg16.parameters():
        p.requires_grad = False

    #KD_opt['feature_dim'] = 512 * 7 ** 2
    KD_opt['feature_dim'] = 1024
    if KD_opt['std'] is None:
        KD_opt['std'] = vgg16.roi_heads.box_predictor.cls_score.weight.data.std()
    if KD_opt['hash_num'] is None:
        KD_opt['hash_num'] = KD_opt['feature_dim'] * 4
    model = FasterRCNNKD(teacher=vgg16, student=vgg11, opt=KD_opt)
    return model

def fasterrcnn_vgg11fpn_with_vgg16fpn(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, 
                            student_ckpt=None, teacher_ckpt=None, KD_opt=None, **kwargs):
    assert KD_opt is not None, 'need opt to define KD loss'
    if trainable_backbone_layers is None:
        trainable_backbone_layers = 3
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0

    s_backbone = vgg_fpn_backbone('vgg11', pretrained=True, returned_layers=[2, 3, 4, 5])
    # In our KD framework, images will be transform by student
    if min_size is None:
        min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    #in mmdetection and detectron2, rpn_pre_nms_top_n_train=12000 rpn_pre_nms_top_n_test=6000
    #in torchvision, rpn_pre_nms_top_n_train=2000 rpn_pre_nms_top_n_test=1000 defaultly.
    #here we use rpn_pre_nms_top_n_train=4000 rpn_pre_nms_top_n_test=2000 to balance the training time and performance
    rpn_pre_nms_top_n_train=4000
    rpn_pre_nms_top_n_test=2000
    vgg11 = FasterRCNN(s_backbone, num_classes, min_size=min_size, 
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test)
    if student_ckpt:
        load_checkpoint(vgg11, ckpt=student_ckpt, teacher=False)

    t_backbone = vgg_fpn_backbone('vgg16', pretrained=False, returned_layers=[2, 3, 4, 5])
    rpn_pre_nms_top_n_train=4000
    rpn_pre_nms_top_n_test=2000
    vgg16 = FasterRCNN(t_backbone, num_classes, min_size=min_size, 
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test)
    load_checkpoint(vgg16, ckpt=teacher_ckpt, teacher=True)

    KD_opt['feature_dim'] = 1024
    model = FasterRCNNKD(teacher=vgg16, student=vgg11, opt=KD_opt)
    return model

if __name__ == '__main__':
    model = faster_rcnn_r50_with_r101(num_classes=91, pretrained_backbone=True)
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)
    import IPython
    IPython.embed()