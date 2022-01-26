import torch
from torch import nn
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection.retinanet import RetinaNet

from torchvision.models.detection.retinanet import RetinaNetHead
from model.retinanet import RetinaNetHeadWithFeature
from model.utils import load_backbone_pretrained
from model.vgg_fpn import vgg_fpn_backbone

def retinanet_vgg11(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, **kwargs):
    load_imagenet = not backbone_pretrained
    backbone = torchvision.models.vgg11(pretrained=load_imagenet)
    load_backbone_pretrained(backbone, backbone_pretrained, 'vgg')
    # the 30th layer of features is relu of conv5_2, remove maxpool
    features = list(backbone.features)[:20]
    '''
    # check default parameters and by default set it to 3 if possible, freeze first 2 blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = 14
    assert trainable_backbone_layers <= 20 and trainable_backbone_layers >= 0
    # freeze top4 conv
    for layer in features[:20 - trainable_backbone_layers]:
        for p in layer.parameters():
            p.requires_grad = False
    '''

    backbone = nn.Sequential(*features)

    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    #head = RetinaNetHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
    head = RetinaNetHeadWithFeature(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
    if min_size is None:
        min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    model = RetinaNet(backbone, num_classes, min_size=min_size,
                    anchor_generator=anchor_generator, head=head)
    return model

def retinanet_vgg11_fpn(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, **kwargs):
    backbone = vgg_fpn_backbone('vgg11', pretrained=True, 
        returned_layers=[3, 4, 5], extra_blocks=LastLevelP6P7(256, 256))
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    #head = RetinaNetHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
    head = RetinaNetHeadWithFeature(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
    if min_size is None:
        min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    model = RetinaNet(backbone, num_classes, min_size=min_size,
                    anchor_generator=anchor_generator, head=head)
    return model

def retinanet_vgg16(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, **kwargs):
    load_imagenet = not backbone_pretrained
    vgg = torchvision.models.vgg16(pretrained=load_imagenet)
    load_backbone_pretrained(vgg, backbone_pretrained, 'vgg')
    # the 30th layer of features is relu of conv5_3
    features = list(vgg.features)[:30]
    '''
    # check default parameters and by default set it to 3 if possible, freeze first 2 blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = 20
    assert trainable_backbone_layers <= 30 and trainable_backbone_layers >= 0
    # freeze top4 conv
    for layer in features[:30 - trainable_backbone_layers]:
        for p in layer.parameters():
            p.requires_grad = False
    #'''
    backbone = nn.Sequential(*features)
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    #head = RetinaNetHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
    head = RetinaNetHeadWithFeature(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
    if min_size is None:
        min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    model = RetinaNet(backbone, num_classes, min_size=min_size,
                    anchor_generator=anchor_generator, head=head)
    return model

def retinanet_vgg16_fpn(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, **kwargs):
    backbone = vgg_fpn_backbone('vgg16', pretrained=True, 
        returned_layers=[3, 4, 5], extra_blocks=LastLevelP6P7(256, 256))
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    #head = RetinaNetHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
    head = RetinaNetHeadWithFeature(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
    if min_size is None:
        min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    model = RetinaNet(backbone, num_classes, min_size=min_size,
                    anchor_generator=anchor_generator, head=head)
    return model


if __name__ == '__main__':
    model = retinanet_vgg11(num_classes=91)
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)
    import IPython
    IPython.embed()