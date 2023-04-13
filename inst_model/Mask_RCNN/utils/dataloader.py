import os.path as osp
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools import mask as coco_mask
from inst_model.Mask_RCNN.utils.utils import cvtColor, preprocess_input
import cv2
from models.transform import Augmentation

def coco_remove_images_without_annotations(dataset, ids):
    """
    删除coco数据集中没有目标，或者目标面积非常小的数据
    refer to:
    https://github.com/pytorch/vision/blob/master/references/detection/coco_utils.py
    :param dataset:
    :param cat_list:
    :return:
    """
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False

        return True

    valid_ids = []
    for ds_idx, img_id in enumerate(ids):
        ann_ids = dataset.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.loadAnns(ann_ids)

        if _has_valid_annotation(anno):
            valid_ids.append(img_id)

    return valid_ids

def convert_coco_poly_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        # 如果mask为空，则说明没有目标，直接返回数值为0的mask
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class MaskDataset(data.Dataset):
    def __init__(self, image_path, coco, COCO_LABEL_MAP={}, augmentation=None,  dataset="train", net_type = None):
        self.img_root     = image_path

        self.coco           = coco
        self.ids            = list(self.coco.imgToAnns.keys())
        
        self.augmentation   = augmentation
        
        self.label_map      = COCO_LABEL_MAP
        self.length         = len(self.ids)

        self._classes = {k: v["name"] for k, v in self.coco.cats.items()}
        self.classes = tuple(self.coco.cats[k]["name"] for k in sorted(self.coco.cats))
    
        # 获取coco数据索引与类别名称的关系
        # 注意在object80中的索引并不是连续的，虽然只有80个类别，但索引还是按照stuff91来排序的
        data_classes = dict([(v["id"], v["name"]) for k, v in self.coco.cats.items()])
        max_index = max(data_classes.keys())  # 90 # len 80
        # 将缺失的类别名称设置成N/A
        coco_classes = {}
        for k in range(1, max_index + 1):
            if k in data_classes:
                coco_classes[k] = data_classes[k]
            else:
                coco_classes[k] = "N/A"    

        self.coco_classes = coco_classes

        ids = list(sorted(self.coco.imgs.keys()))
        if dataset == "train":
            # 移除没有目标，或者目标面积非常小的数据
            valid_ids = coco_remove_images_without_annotations(self.coco, ids)
            self.ids = valid_ids
        else:
            self.ids = ids
        
        # self.label_map = label_map
        self.net_type = net_type

    def parse_targets(self,
                      img_id: int,
                      coco_targets: list,
                      w: int = None,
                      h: int = None):
        assert w > 0
        assert h > 0

        # 只筛选出单个对象的情况
        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # classes = [obj["category_id"] for obj in anno]
        classes = [self.label_map[obj["category_id"]] -1 for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_mask(segmentations, h, w)

        # 筛选出合法的目标，即x_max>x_min且y_max>y_min
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        area = area[keep]
        iscrowd = iscrowd[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target
 
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_root, path)).convert('RGB')

        w, h = img.size
        target = self.parse_targets(img_id, coco_target, w, h)
        

        # if True:
        if len(target["area"]) == 0: return None, None, None, None
        img = cv2.imread(os.path.join(self.img_root, path))
        img       = np.array(img, np.float32)
        height, width, _ = img.shape

        masks = target["masks"].numpy().astype(np.float32)

        num_crowds = sum([x for x in target['iscrowd']])

        boxes_classes = []                              
        for i in range(len(target["labels"])):
            bbox        = target['boxes'][i]
            final_box   = [bbox[0], bbox[1], 0 + bbox[2], 0 + bbox[3], target['labels'][i] - 1]
            boxes_classes.append(final_box)
        boxes_classes = np.array(boxes_classes, np.float32)

        # print(type(boxes_classes), len(boxes_classes), height, width)
        if len(boxes_classes) > 1:
            boxes_classes[:, [0, 2]] /= width
            boxes_classes[:, [1, 3]] /= height   
        else:
            boxes_classes[0, [0, 2]] /= width
            boxes_classes[0, [1, 3]] /= height   

        img, masks, boxes, labels = self.augmentation(img, masks, boxes_classes[:, :4], {'num_crowds': num_crowds, 'labels': boxes_classes[:, 4]})
        num_crowds  = labels['num_crowds']
        labels      = labels['labels']
        boxes       = np.concatenate([boxes, np.expand_dims(labels, axis=1)], -1) # 0123 x y w n

        img = img.astype(np.uint8)           

        # boxes[:, [0, 2]] *= 544
        # boxes[:, [1, 3]] *= 544 

        # for b in boxes:                                 
        #     cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255,255,0), 2) # 邊框
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        image = preprocess_input(img)
        image = np.transpose(image, [2, 0, 1])
        image = torch.from_numpy(np.array(image, np.float32))

        labels = boxes[:, -1]        

        target = dict(image_id=torch.tensor([img_id], dtype=torch.int64), 
                    boxes=torch.tensor(boxes[:, :-1], dtype=torch.float32), 
                    labels=torch.tensor(boxes[:, -1], dtype=torch.int64), 
                    masks=torch.tensor(masks, dtype=torch.uint8))
        return image, target
        

    def __len__(self):
        return len(self.ids)

    def get_height_and_width(self, index):
        coco = self.coco
        img_id = self.ids[index]

        img_info = coco.loadImgs(img_id)[0]
        w = img_info["width"]
        h = img_info["height"]
        return h, w 

def mask_dataset_collate(batch):
    images      = []
    targets     = []
    for sample in batch:
        if isinstance(sample[0], type(None)): continue
        # print(sample[0])
        # print(type(sample[0]), type(sample[0]) is None, type(sample[0]) == None, isinstance(sample[0], type(None)))
        images.append(sample[0])
        targets.append(sample[1])
    return images, targets