import torch
from torch import nn
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection.retinanet import RetinaNet

from model.utils import load_backbone_pretrained, load_det_checkpoint
from model.retinanet import RetinaNetHeadWithFeature
from model.KD import create_KD_loss
from model.vgg_fpn import vgg_fpn_backbone

class RetinaNetKD(nn.Module):
    """
    Main class for RetinaNet KD
    """
    def __init__(self, teacher, student, opt=None):
        super(RetinaNetKD, self).__init__()
        # the teacher will not update
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        self.t_transform = teacher.transform
        self.t_backbone = teacher.backbone
        self.t_anchor_generator = teacher.anchor_generator
        self.t_head = teacher.head
        
        self.s_transform = student.transform
        self.s_backbone = student.backbone
        self.s_anchor_generator = student.anchor_generator
        self.s_head = student.head

        self.postprocess_detections = student.postprocess_detections
        self.proposal_matcher = student.proposal_matcher

        if opt['hash_num'] is None:
            opt['hash_num'] = opt['feature_dim'] * 4
        if opt['std'] is None:
            opt['std'] = self.t_head.classification_head.cls_logits.weight.data.std()
        self.feat_KD_loss = create_KD_loss(opt)


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

        images, targets = self.s_transform(images, targets)
  
        # only compute student
        s_feature = self.s_backbone(images.tensors)
        if isinstance(s_feature, torch.Tensor):
            s_feature = OrderedDict([('0', s_feature)])
        s_feature = list(s_feature.values())
        # compute the retinanet heads outputs using the features
        s_head_outputs = self.s_head(s_feature, return_feat=self.training)
        # create the set of anchors
        s_anchors = self.s_anchor_generator(images, s_feature)

        losses = {}
        s_detections = torch.jit.annotate(List[Dict[str, Tensor]], [])
        # eval the student
        if not self.training:
            # compute the detections
            s_detections = self.postprocess_detections(s_head_outputs, s_anchors, images.image_sizes)
            s_detections = self.s_transform.postprocess(s_detections, images.image_sizes, original_image_sizes)
            return s_detections

        # train
        assert targets is not None 
        with torch.no_grad():
            t_feature = self.t_backbone(images.tensors)
            if isinstance(t_feature, torch.Tensor):
                t_feature = OrderedDict([('0', t_feature)])
            t_feature = list(t_feature.values())
            t_head_outputs = self.t_head(t_feature, return_feat=True)  

        losses = self.compute_loss(targets, s_head_outputs, t_head_outputs, s_anchors)

        return losses


    def compute_loss(self, targets, s_head_outputs, t_head_outputs, s_anchors):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(s_anchors, targets):
            if targets_per_image['boxes'].numel() == 0:
                matched_idxs.append(torch.empty((0,), dtype=torch.int32))
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image['boxes'], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        loss = self.s_head.compute_loss(targets, s_head_outputs, s_anchors, matched_idxs)

        s_cls_feat = s_head_outputs['cls_feat']
        t_cls_feat = t_head_outputs['cls_feat']

        # KD loss
        kd_loss = torch.tensor(0., device=s_cls_feat.device)
        feat_num = 0
        for i in range(len(matched_idxs)):
            matched_idx = matched_idxs[i]
            kd_mask = matched_idx.reshape(-1, 9)
            kd_mask = (kd_mask.sum(1) > -2 * 9).detach()
            if kd_mask.sum() == 0:
                continue
            feat_num += 1
            kd_loss += self.feat_KD_loss(s_cls_feat[i, kd_mask], t_cls_feat[i, kd_mask])
        kd_loss /= feat_num + 1e-7

        kd_losses = {'kd_loss': kd_loss}
        loss.update(kd_losses)

        return loss



def retinanet_r50_with_r101(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, 
                            student_ckpt=None, teacher_ckpt=None, KD_opt=None, **kwargs):
    if student_ckpt is None or not backbone_pretrained:
        load_imagenet = True
    else:
        load_imagenet = False
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = resnet_fpn_backbone('resnet50', load_imagenet, 
                                returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256))
    if backbone_pretrained:
        load_backbone_pretrained(backbone, backbone_pretrained, 'resnet_fpn')
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    head = RetinaNetHeadWithFeature(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
    if min_size is None:
        min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    r50 = RetinaNet(backbone, num_classes, min_size=min_size,
                    anchor_generator=anchor_generator, head=head)
    if student_ckpt:
        load_det_checkpoint(r50, ckpt=student_ckpt, teacher=False)

    # skip P2 because it generates too many anchors (according to their paper)
    backbone = resnet_fpn_backbone('resnet101', False, 
                                returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256))
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    head = RetinaNetHeadWithFeature(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
    r101 = RetinaNet(backbone, num_classes, min_size=min_size,
                    anchor_generator=anchor_generator, head=head)
    load_det_checkpoint(r101, ckpt=teacher_ckpt, teacher=True)

    KD_opt['feature_dim'] = 256 * 3 * 3
    model = RetinaNetKD(teacher=r101, student=r50, opt=KD_opt)
    return model


def retinanet_vgg11fpn_with_vgg16fpn(num_classes=91, min_size=None, backbone_pretrained='', trainable_backbone_layers=None, 
                            student_ckpt=None, teacher_ckpt=None, KD_opt=None, **kwargs):
    if student_ckpt is None or not backbone_pretrained:
        load_imagenet = True
    else:
        load_imagenet = False
    backbone = vgg_fpn_backbone('vgg11', pretrained=True, 
        returned_layers=[3, 4, 5], extra_blocks=LastLevelP6P7(256, 256))
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    head = RetinaNetHeadWithFeature(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
    if min_size is None:
        min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    vgg11 = RetinaNet(backbone, num_classes, min_size=min_size,
                    anchor_generator=anchor_generator, head=head)
    if student_ckpt:
        load_det_checkpoint(vgg11, ckpt=student_ckpt, teacher=False)

    backbone = vgg_fpn_backbone('vgg16', pretrained=False, 
        returned_layers=[3, 4, 5], extra_blocks=LastLevelP6P7(256, 256))
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    head = RetinaNetHeadWithFeature(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
    vgg16 = RetinaNet(backbone, num_classes, min_size=min_size,
                    anchor_generator=anchor_generator, head=head)
    load_det_checkpoint(vgg16, ckpt=teacher_ckpt, teacher=True)

    KD_opt['feature_dim'] = 256 * 3 * 3
    model = RetinaNetKD(teacher=vgg16, student=vgg11, opt=KD_opt)
    return model

if __name__ == '__main__':
    model = retinanet_r50_with_r101(num_classes=91)
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)
    import IPython
    IPython.embed()