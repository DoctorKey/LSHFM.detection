from dataset.coco_utils import get_coco
from dataset.voc_utils import get_voc07, get_voc0712

dataset_dict = {
    "coco": get_coco,
    "voc07": get_voc07,
    "voc0712": get_voc0712,
}