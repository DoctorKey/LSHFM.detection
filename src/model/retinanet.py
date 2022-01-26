import math
from collections import OrderedDict

import torch
from torch import nn
from torch import Tensor
from torch.jit.annotations import Dict, List, Tuple

from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNetClassificationHead, RetinaNetRegressionHead


    
class RetinaNetHeadWithFeature(RetinaNetHead):
    """
    A regression and classification head for use in RetinaNet.

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """
    def __init__(self, in_channels, num_anchors, num_classes):
        super(RetinaNetHeadWithFeature, self).__init__(in_channels, num_anchors, num_classes)
        self.classification_head = RetinaNetClassificationHeadWithFeature(in_channels, num_anchors, num_classes)
        self.regression_head = RetinaNetRegressionHeadWithFeature(in_channels, num_anchors)

    def forward(self, x, return_feat=False):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        cls_feat, cls_logits = self.classification_head(x)
        bbox_feat, bbox_reg = self.regression_head(x)
        result = dict()
        result['cls_logits'] = cls_logits
        result['bbox_regression'] = bbox_reg
        if return_feat:
            result['cls_feat'] = cls_feat
            result['bbox_feat'] = bbox_feat
        return result

class RetinaNetClassificationHeadWithFeature(RetinaNetClassificationHead):
    """
    A classification head for use in RetinaNet.

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """
    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_feat = []
        all_cls_logits = []

        for features in x:
            cls_feat = self.conv(features)
            cls_logits = self.cls_logits(cls_feat)

            cls_feat = torch.nn.functional.unfold(cls_feat, (3,3), padding=1)
            cls_feat = cls_feat.transpose(1, 2)
            all_cls_feat.append(cls_feat)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_feat, dim=1), torch.cat(all_cls_logits, dim=1)


class RetinaNetRegressionHeadWithFeature(RetinaNetRegressionHead):
    """
    A regression head for use in RetinaNet.

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """
    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_bbox_feat = []
        all_bbox_regression = []

        for features in x:
            bbox_feat = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_feat)

            # bbox_feat (N, HW, C * 3 * 3)
            bbox_feat = torch.nn.functional.unfold(bbox_feat, (3,3), padding=1)
            bbox_feat = bbox_feat.transpose(1, 2)
            all_bbox_feat.append(bbox_feat)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_feat, dim=1), torch.cat(all_bbox_regression, dim=1)


