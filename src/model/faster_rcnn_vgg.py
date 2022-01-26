import torch
from torch import nn
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
#from torchvision.models.detection.faster_rcnn import FasterRCNN

from model.faster_rcnn import FasterRCNN
from model.utils import load_backbone_pretrained
from model.vgg_fpn import vgg_fpn_backbone

class VGGRoiHead(nn.Module):
    """docstring for VGGRoiHead"""
    def __init__(self, vgg, representation_size=4096):
        super(VGGRoiHead, self).__init__()
        self.fc1 = vgg.classifier[:2]
        self.fc2 = vgg.classifier[3:5]
        #self.fc = torch.nn.Sequential(*vgg.classifier[:5])
        self.representation_size = representation_size

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def fasterrcnn_vgg11(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, **kwargs):
    """
    Constructs a Faster R-CNN model with a VGG11 backbone.
    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction
    Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.
    Example::
        >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        >>> # For training
        >>> images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
        >>> labels = torch.randint(1, 91, (4, 11))
        >>> images = list(image for image in images)
        >>> targets = []
        >>> for i in range(len(images)):
        >>>     d = {}
        >>>     d['boxes'] = boxes[i]
        >>>     d['labels'] = labels[i]
        >>>     targets.append(d)
        >>> output = model(images, targets)
        >>> # For inference
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)
    Arguments:
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        num_classes (int): number of output classes of the model (including the background)
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    # load a pre-trained model for classification and return
    load_imagenet = not backbone_pretrained
    backbone = torchvision.models.vgg11(pretrained=load_imagenet)
    load_backbone_pretrained(backbone, backbone_pretrained, 'vgg')
    # the 30th layer of features is relu of conv5_2, remove maxpool
    features = list(backbone.features)[:20]

    # check default parameters and by default set it to 3 if possible, freeze first 2 blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = 14
    assert trainable_backbone_layers <= 20 and trainable_backbone_layers >= 0
    # freeze top4 conv
    for layer in features[:20 - trainable_backbone_layers]:
        for p in layer.parameters():
            p.requires_grad = False

    backbone = nn.Sequential(*features)

    # FasterRCNN needs to know the number of
    # output channels in a backbone. For vgg16, it's 512
    # so we need to add it here
    backbone.out_channels = 512

    # let's make the RPN generate 3 x 3 anchors per spatial
    # location, with 3 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    roi_head = None

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
                                                    output_size=7,
                                                    sampling_ratio=2)

    if min_size is None:
        min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    rpn_pre_nms_top_n_train=4000
    rpn_post_nms_top_n_train=2000
    rpn_pre_nms_top_n_test=2000
    rpn_post_nms_top_n_test=1000
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       min_size=min_size,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler, box_head=roi_head,
                       rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                    rpn_post_nms_top_n_train=rpn_post_nms_top_n_train, rpn_post_nms_top_n_test=rpn_post_nms_top_n_test)
    return model

def fasterrcnn_vgg11_fpn(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, **kwargs):
    backbone = vgg_fpn_backbone('vgg11', pretrained=True, returned_layers=[2, 3, 4, 5])
    if min_size is None:
        min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    #in mmdetection and detectron2, rpn_pre_nms_top_n_train=12000 rpn_pre_nms_top_n_test=6000
    #in torchvision, rpn_pre_nms_top_n_train=2000 rpn_pre_nms_top_n_test=1000 defaultly.
    #here we use rpn_pre_nms_top_n_train=4000 rpn_pre_nms_top_n_test=2000 to balance the training time and performance
    rpn_pre_nms_top_n_train=4000
    rpn_post_nms_top_n_train=2000
    rpn_pre_nms_top_n_test=2000
    rpn_post_nms_top_n_test=1000
    model = FasterRCNN(backbone, num_classes, min_size=min_size, 
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train, rpn_post_nms_top_n_test=rpn_post_nms_top_n_test)
    return model

def fasterrcnn_vgg16(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, **kwargs):
    """
    Constructs a Faster R-CNN model with a VGG16 backbone.
    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction
    Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.
    Example::
        >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        >>> # For training
        >>> images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
        >>> labels = torch.randint(1, 91, (4, 11))
        >>> images = list(image for image in images)
        >>> targets = []
        >>> for i in range(len(images)):
        >>>     d = {}
        >>>     d['boxes'] = boxes[i]
        >>>     d['labels'] = labels[i]
        >>>     targets.append(d)
        >>> output = model(images, targets)
        >>> # For inference
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)
    Arguments:
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        num_classes (int): number of output classes of the model (including the background)
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    # load a pre-trained model for classification and return
    load_imagenet = not backbone_pretrained
    vgg = torchvision.models.vgg16(pretrained=load_imagenet)
    load_backbone_pretrained(vgg, backbone_pretrained, 'vgg')
    # the 30th layer of features is relu of conv5_3
    features = list(vgg.features)[:30]

    # check default parameters and by default set it to 3 if possible, freeze first 2 blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = 20
    assert trainable_backbone_layers <= 30 and trainable_backbone_layers >= 0
    # freeze top4 conv
    for layer in features[:30 - trainable_backbone_layers]:
        for p in layer.parameters():
            p.requires_grad = False

    backbone = nn.Sequential(*features)

    # FasterRCNN needs to know the number of
    # output channels in a backbone. For vgg16, it's 512
    # so we need to add it here
    backbone.out_channels = 512

    # let's make the RPN generate 3 x 3 anchors per spatial
    # location, with 3 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    #roi_head = vgg.classifier[:5]
    #roi_head = torch.nn.Sequential(*vgg.classifier[:6])
    #roi_head.representation_size = 4096
    roi_head = None
    #roi_head = VGGRoiHead(vgg, 4096)

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
                                                    output_size=7,
                                                    sampling_ratio=2)

    if min_size is None:
        min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    rpn_pre_nms_top_n_train=4000
    rpn_post_nms_top_n_train=2000
    rpn_pre_nms_top_n_test=2000
    rpn_post_nms_top_n_test=1000
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       min_size=min_size,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler, box_head=roi_head,
                       rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                    rpn_post_nms_top_n_train=rpn_post_nms_top_n_train, rpn_post_nms_top_n_test=rpn_post_nms_top_n_test)
    return model

def fasterrcnn_vgg16_fpn(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, **kwargs):
    backbone = vgg_fpn_backbone('vgg16', pretrained=True, returned_layers=[2, 3, 4, 5])
    if min_size is None:
        min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    #in mmdetection and detectron2, rpn_pre_nms_top_n_train=12000 rpn_pre_nms_top_n_test=6000
    #in torchvision, rpn_pre_nms_top_n_train=2000 rpn_pre_nms_top_n_test=1000 defaultly.
    #here we use rpn_pre_nms_top_n_train=4000 rpn_pre_nms_top_n_test=2000 to balance the training time and performance
    rpn_pre_nms_top_n_train=4000
    rpn_post_nms_top_n_train=2000
    rpn_pre_nms_top_n_test=2000
    rpn_post_nms_top_n_test=1000
    model = FasterRCNN(backbone, num_classes, min_size=min_size, 
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train, rpn_post_nms_top_n_test=rpn_post_nms_top_n_test)
    return model

if __name__ == '__main__':
    model = fasterrcnn_vgg11(num_classes=91, pretrained_backbone=True)
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    #predictions = model(x)
    import IPython
    IPython.embed()