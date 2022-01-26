'''
this file is based on https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/utils/vis_tool.py
'''
import time

import numpy as np
import matplotlib
import torch

#matplotlib.use('Agg')
from matplotlib import pyplot as plt

from dataset import dataset_dict
from model import model_dict


def vis_image(img, ax=None):
    """Visualize a color image.
    Args:
        img (numpy.ndarray or torch.tensor): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]` or :math:`[0, 1]`.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.
    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.
    """
    if 'torch' in str(type(img)):
        img = img.cpu().numpy()
    if img.max() <= 1.0:
        img = img * 255
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    # CHW -> HWC
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))
    return ax

def vis_bbox(img, bbox, label=None, score=None, ax=None, vis_label=None):
    """Visualize bounding boxes inside image.
    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        bbox (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :math:`[x0, y0, x1, y1]` in the second axis.
        label (~numpy.ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        label_names (iterable of strings): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.
    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.
    """

    label_names = vis_label
    if label is not None and vis_label is None:
        raise ValueError('must exist vis_label for display')
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    for i, bb in enumerate(bbox):
        xy = (bb[0], bb[1])
        height = bb[3] - bb[1]
        width = bb[2] - bb[0]
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))

        caption = list()

        if label is not None and label_names is not None:
            lb = label[i]
            if not (0 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[0], bb[1],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax

def dispaly_ground_truth(dataset, idx):
    data = dataset[idx]
    img = data[0]
    labels = data[1]["labels"]
    bbox = data[1]['boxes']
    vis_bbox(img, bbox, label=labels, vis_label=dataset.vis_label)
    plt.show()

def display_model_predict(dataset, idx, model, checkpoint):
    data = dataset[idx]
    img = data[0]
    ckpt = torch.load(checkpoint)
    state_dict = ckpt['model']
    ret = model.load_state_dict(state_dict)
    print(ret)
    model.eval()
    with torch.no_grad():
        output = model(img.unsqueeze(0).cuda())
    output = output[0]
    boxes = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    vis_bbox(img, boxes, label=labels, score=scores, vis_label=dataset.vis_label)
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--dataset', default='voc07', help='dataset')
    parser.add_argument("--test", dest="test", help="use test dataset", action="store_true")
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model', 
        choices=['fasterrcnn_resnet101_fpn', 'fasterrcnn_resnet50_fpn', 
                'faster_rcnn_vgg16', 'fasterrcnn_vgg11',
                'faster_rcnn_r50_with_r101',
                'retinanet_resnet50_fpn'])
    parser.add_argument('--idx', default=0, type=int,
                        help='idx data for dataset')
    parser.add_argument('--checkpoint', default='', help='path of checkpoint')
    parser.add_argument(
        "--gt",
        dest="gt",
        help="Only show ground truth",
        action="store_true",
    )

    args = parser.parse_args()

    dataset, dataset_test, num_classes = dataset_dict[args.dataset]()
    if args.test:
        dataset = dataset_test
    if args.gt:
        dispaly_ground_truth(dataset, args.idx)
    else:
        model = model_dict[args.model](num_classes=num_classes, pretrained=False)
        model = model.cuda()

        display_model_predict(dataset, args.idx, model, args.checkpoint)

