import copy
import os
from PIL import Image
import socket

import torch
import torch.utils.data
import torchvision

import dataset.transforms as T

VOC_CLASS_LIST = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

VOC_VIS_LABEL_NAMES = (
    'bg',
    'fly',
    'bike',
    'bird',
    'boat',
    'pin',
    'bus',
    'c',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'moto',
    'p',
    'plant',
    'shep',
    'sofa',
    'train',
    'tv',
)

def get_data_path():
    hostname = socket.gethostname()
    if 'amax' == hostname:
        # M40
        root = '/opt/Dataset/VOCdevkit'
    elif 'Pascal2' in hostname:
        root = '/opt/Dataset/VOCdevkit'
    elif 'Pascal1' in hostname:
        root = '/opt/Dataset/VOCdevkit'
    elif 'ubuntu' in hostname:
        # V100
        root = '/root/Dataset/VOC'
    elif 'admin.cluster' in hostname:
        # 2080Ti
        root = '/amax/opt/Dataset'
    elif 'GPU4' in hostname:
        #root = '/opt/Dataset/VOC'
        root = '/mnt/ramdisk/VOC'
    elif 'gpu3' in hostname:
        root = '/opt/Dataset'
    elif 'digix' in hostname:
        root = '/mnt/ramdisk/VOC'
    return root



class ConvertToCocoAnno(object):
    def __init__(self, ignore_difficult=True):
        self._ignore_difficult = ignore_difficult
    '''
    image: a PIL Image of size (H, W)
    target: a dict containing the following fields
        boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
        labels (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.
        image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
        area (Tensor[N]): The area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
        iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
    '''
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target['annotations']['annotation']['object']

        if self._ignore_difficult:
            anno = [obj for obj in anno if obj['difficult'] == '0']

        boxes = []
        for obj in anno:
            bndbox = obj["bndbox"]
            bbox = [int(bndbox[x]) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1
            bbox[1] -= 1
            boxes.append(bbox)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [VOC_CLASS_LIST.index(obj["name"]) for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        #area = torchvision.ops.box_area(boxes)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.tensor([int(obj["difficult"]) for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target



class VOCDetection(torchvision.datasets.VOCDetection):
    def __init__(self, root, year, image_set, transforms):
        super(VOCDetection, self).__init__(root, year, image_set)
        self._transforms = transforms
        self._offset = 0
        self.vis_label = VOC_VIS_LABEL_NAMES

    def __getitem__(self, idx):
        img, target = super(VOCDetection, self).__getitem__(idx)
        image_id = self._get_ids(idx)
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def _get_ids(self, idx):
        return idx + self._offset

    def set_idx_offset(self, offset):
        """
            this method used for concat dataset
        """
        self._offset = offset


def get_voc07():
    root = get_data_path()

    t = [ConvertToCocoAnno(ignore_difficult=True), T.ToTensor(), T.RandomHorizontalFlip(0.5)]
    train_transforms = T.Compose(t)
    train_dataset = VOCDetection(root, '2007', 'trainval', transforms=train_transforms)

    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    t = [ConvertToCocoAnno(ignore_difficult=True), T.ToTensor()]
    test_transforms = T.Compose(t)
    test_dataset = VOCDetection(root, '2007', 'test', transforms=test_transforms)

    num_classes = 21
    # train: 5011
    # test: 4952
    return train_dataset, test_dataset, num_classes


def get_voc0712():
    root = get_data_path()
    # We find that it is better to ignore the difficult anno
    t = [ConvertToCocoAnno(ignore_difficult=True), T.ToTensor(), T.RandomHorizontalFlip(0.5)]
    train_transforms = T.Compose(t)
    train07_dataset = VOCDetection(root, '2007', 'trainval', transforms=train_transforms)
    train12_dataset = VOCDetection(root, '2012', 'trainval', transforms=train_transforms)
    # idx for train12 from len(train07_dataset)
    train12_dataset.set_idx_offset(len(train07_dataset))
    train_dataset = torch.utils.data.ConcatDataset([train07_dataset, train12_dataset])
    train_dataset.vis_label = VOC_VIS_LABEL_NAMES

    t = [ConvertToCocoAnno(ignore_difficult=True), T.ToTensor()]
    test_transforms = T.Compose(t)
    test_dataset = VOCDetection(root, '2007', 'test', transforms=test_transforms)

    num_classes = 21
    # train: 16551
    # test: 4952
    return train_dataset, test_dataset, num_classes

if __name__ == '__main__':
    train_dataset, test_dataset, num_classes = get_voc07()
    import IPython
    IPython.embed()

