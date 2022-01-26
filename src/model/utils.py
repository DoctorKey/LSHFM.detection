import os
import torch

def load_backbone_pretrained(model, pretrained, arch):
    if not pretrained:
        return model

    print("=> loading pretrained from checkpoint {}".format(pretrained))
    if os.path.isfile(pretrained):
        checkpoint = torch.load(pretrained, map_location='cpu')
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('module.', ''): v for k,v in state_dict.items()}
    elif pretrained.startswith('http'):
        state_dict = torch.hub.load_state_dict_from_url(pretrained, map_location='cpu')

    if 'resnet_c4' in arch:
        ret = model.load_state_dict(state_dict, strict=False)
        print("=> load result {} ".format(ret))
    elif 'resnet_fpn' in arch:
        state_dict = {'body.{}'.format(k): v for k,v in state_dict.items()}
        ret = model.load_state_dict(state_dict, strict=False)
        print("=> load result {} ".format(ret))
    elif 'vgg' in arch:
        #state_dict = {k.replace('features.', ''): v for k,v in state_dict.items()}
        ret = model.load_state_dict(state_dict, strict=False)
        print("=> load result {} ".format(ret))
    elif 'with' in arch:
        # KD
        pass
    return model

def load_det_checkpoint(model, ckpt=None, teacher=True):
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

def fix_BN_stat(module):
    classname = module.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        module.eval()
