import torch
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
#from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from model.faster_rcnn import FasterRCNN
from model.utils import load_backbone_pretrained

def fasterrcnn_resnet50_c4(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-50-C4 backbone.
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
    Arguments:
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        num_classes (int): number of output classes of the model (including the background)
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    # check default parameters and by default set it to 3 if possible
    if trainable_backbone_layers is None:
        trainable_backbone_layers = 3
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0

    load_imagenet = not backbone_pretrained
    resnet50 = torchvision.models.resnet50(pretrained=load_imagenet)
    load_backbone_pretrained(resnet50, backbone_pretrained, 'resnet_c4')
    features = [resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool, 
            resnet50.layer1, resnet50.layer2, resnet50.layer3]
    backbone = torch.nn.Sequential(*features)
    backbone.out_channels = 1024
    # freeze backbone
    for layer in backbone[:8 - trainable_backbone_layers]:
        for p in layer.parameters():
            p.requires_grad = False

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    roi_head = torch.nn.Sequential(resnet50.layer4, resnet50.avgpool)
    roi_head.representation_size = 2048
    # we find that it is better to output 7x7 than 14x14
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
                                                    output_size=7,
                                                    sampling_ratio=2)
    if min_size is None:
        min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    rpn_pre_nms_top_n_train=4000
    rpn_post_nms_top_n_train=2000
    rpn_pre_nms_top_n_test=2000
    rpn_post_nms_top_n_test=1000
    model = FasterRCNN(backbone, num_classes=num_classes, min_size=min_size, rpn_anchor_generator=anchor_generator,
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train, rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        box_roi_pool=roi_pooler, box_head=roi_head)
    return model

def fasterrcnn_resnet101_c4(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-101-C4 backbone.
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
    Arguments:
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        num_classes (int): number of output classes of the model (including the background)
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    # check default parameters and by default set it to 3 if possible
    if trainable_backbone_layers is None:
        trainable_backbone_layers = 3
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0

    load_imagenet = not backbone_pretrained
    resnet = torchvision.models.resnet101(pretrained=load_imagenet)
    load_backbone_pretrained(resnet, backbone_pretrained, 'resnet_c4')
    features = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, 
            resnet.layer1, resnet.layer2, resnet.layer3]
    backbone = torch.nn.Sequential(*features)
    backbone.out_channels = 1024
    # freeze backbone
    for layer in backbone[:8 - trainable_backbone_layers]:
        for p in layer.parameters():
            p.requires_grad = False

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    roi_head = torch.nn.Sequential(resnet.layer4, resnet.avgpool)
    roi_head.representation_size = 2048
    # we find that it is better to output 7x7 than 14x14
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
                                                    output_size=7,
                                                    sampling_ratio=2)
    if min_size is None:
        min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    rpn_pre_nms_top_n_train=4000
    rpn_post_nms_top_n_train=2000
    rpn_pre_nms_top_n_test=2000
    rpn_post_nms_top_n_test=1000
    model = FasterRCNN(backbone, num_classes=num_classes, min_size=min_size, rpn_anchor_generator=anchor_generator,
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train, rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        box_roi_pool=roi_pooler, box_head=roi_head)
    return model

def fasterrcnn_resnet50_fpn(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.
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
    # check default parameters and by default set it to 3 if possible
    if trainable_backbone_layers is None:
        trainable_backbone_layers = 3
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0

    load_imagenet = not backbone_pretrained
    backbone = resnet_fpn_backbone('resnet50', load_imagenet, trainable_layers=trainable_backbone_layers)
    load_backbone_pretrained(backbone, backbone_pretrained, 'resnet_fpn')
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
    
def fasterrcnn_resnet101_fpn(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-101-FPN backbone.
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
    # check default parameters and by default set it to 3 if possible, freeze first 2 blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = 3
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0

    load_imagenet = not backbone_pretrained
    backbone = resnet_fpn_backbone('resnet101', load_imagenet, trainable_layers=trainable_backbone_layers)
    load_backbone_pretrained(backbone, backbone_pretrained, 'resnet_fpn')
    if min_size is None:
        min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    rpn_pre_nms_top_n_train=4000
    rpn_post_nms_top_n_train=2000
    rpn_pre_nms_top_n_test=2000
    rpn_post_nms_top_n_test=1000
    model = FasterRCNN(backbone, num_classes, min_size=min_size, 
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train, rpn_post_nms_top_n_test=rpn_post_nms_top_n_test)
    return model

if __name__ == '__main__':
    model = fasterrcnn_resnet50_c4(num_classes=91)
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)
    import IPython
    IPython.embed()