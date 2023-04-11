import sys
sys.path.append("../")
sys.path.append(".")
import os
import random
from pycocotools.coco import COCO
from utils.helpers import get_classes

def get_coco_label_map(coco, class_names):
    COCO_LABEL_MAP = {}

    coco_cat_index_MAP = {}
    for index, cat in coco.cats.items():
        if cat['name'] == '_background_':
            continue
        coco_cat_index_MAP[cat['name']] = index

    for index, class_name in enumerate(class_names):
        COCO_LABEL_MAP[coco_cat_index_MAP[class_name]] = index + 1
    return COCO_LABEL_MAP

def get_annotation(data_root, classes_path):
    random.seed(0)
    print("Generate txt in ImageSets.")

    #---------------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    # classes_path    = 'model_data/{}_classes.txt' %(data_name)
    class_names, num_classes = get_classes(classes_path)
    

    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    VOCdevkit_path  = os.path.join(data_root, "VOCdevkit")

    train_image_path        = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages")
    val_image_path          = os.path.join(VOCdevkit_path,"VOC2012/JPEGImages")

    train_coco  = COCO(os.path.join(VOCdevkit_path, "VOC2007/Annotations/VOC2007.json"))
    val_coco    = COCO(os.path.join(VOCdevkit_path, "VOC2012/Annotations/VOC2012.json")) 

    COCO_LABEL_MAP  = get_coco_label_map(train_coco, class_names)
    return train_image_path, val_image_path, train_coco, val_coco, \
        class_names, num_classes, COCO_LABEL_MAP


if __name__ == "__main__":
    data_root = '/home/leyan/DataSet/'
    get_annotation(data_root) 