import torch
from torch import nn
import torchvision

from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection.backbone_utils import BackboneWithFPN

cfgs = {
    'vgg11': [3, 6, 11, 16],
    'vgg16': [5, 10, 17, 24],
}

class VGGWithFeature(nn.Module):
    """docstring for VGGWithFeature"""
    def __init__(self, name, pretrained=False):
        super(VGGWithFeature, self).__init__()
        assert name in cfgs.keys(), 'Only support {}'.format(cfgs.keys())
        vgg = torchvision.models.__dict__[name](pretrained=pretrained)
        features = vgg.features
        cfg = cfgs[name]
        self.layer1 = nn.Sequential(features[:cfg[0]])
        self.layer2 = nn.Sequential(features[cfg[0]:cfg[1]])
        self.layer3 = nn.Sequential(features[cfg[1]:cfg[2]])
        self.layer4 = nn.Sequential(features[cfg[2]:cfg[3]])
        self.layer5 = nn.Sequential(features[cfg[3]:])
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def vgg_fpn_backbone(
    backbone_name,
    pretrained,
    trainable_layers=3,
    returned_layers=None,
    extra_blocks=None
):
    backbone = VGGWithFeature(backbone_name, pretrained=pretrained)
    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['layer5', 'layer4', 'layer3', 'layer2', 'layer1'][:trainable_layers]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4, 5]
    assert min(returned_layers) > 0 and max(returned_layers) < 6
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    in_channels_list = [64, 128, 256, 512, 512]
    in_channels_list = [in_channels_list[i-1] for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)



if __name__ == '__main__':
    vgg = VGGWithFeature('vgg16')
    x = torch.randn(1, 3, 224, 224)
    fpn = vgg_fpn_backbone('vgg11', False)
    import IPython
    IPython.embed()