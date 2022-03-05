
from matplotlib.pyplot import axis
# from pyrsistent import T
import torch
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from PIL import Image

from torch.utils.data import Dataset

from pycocotools.coco import COCO
import cv2
import numpy as np

import albumentations as A
import random


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y    
  

class Albumentations:
    #  Albumentations class (optional, only used if package is installed)
    def __init__(self, input_shape, train):
        self.transform = None
        try:           
            if train:
                self.transform = A.Compose([                  
                    A.Resize(input_shape[0], input_shape[1]),
                    A.Blur(p=0.1),
                    A.MedianBlur(p=0.1),
                    A.ToGray(p=0.1),
                    A.CLAHE(p=0.1),
                    A.RandomBrightnessContrast(p=0.0),
                    A.RandomGamma(p=0.0),
                    A.ImageCompression(quality_lower=75, p=0.0),
                    A.HorizontalFlip(p=0.1),
                    A.VerticalFlip(p=0.1)

                    # img = (img - mean * max_pixel_value) / (std * max_pixel_value)
                    ,A.transforms.Normalize(
                        mean = (0.485, 0.456, 0.406), 
                        std = (0.229, 0.224, 0.225))
                    ],
                    bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'])
                    )
            else:
                self.transform = A.Compose([
                    A.Resize(input_shape[0], input_shape[1])    
                    ,A.transforms.Normalize(
                        mean = (0.485, 0.456, 0.406), 
                        std = (0.229, 0.224, 0.225))                
                    ],
                    bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'])
                    )

            # LOGGER.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass      

    def __call__(self, image, boxes, labels, masks, p=1.0):
        if self.transform and random.random() < p:

            new = self.transform(image=image, bboxes=boxes, class_labels=labels, mask=masks)  # transformed
            img = new['image']
            # boxs = new['bboxes']
            # labels = new['class_labels']
            mask = new["mask"]

            labels = np.array([[*c, b] for c, b in zip(new['bboxes'], new['class_labels'])])
            boxs = labels[:, :4]
            cls_labels = labels[:, 4]

            # im, labels, mask = new['image'], np.array([[*c, b] for c, b in zip(new['bboxes'], new['class_labels'])]), new["mask"]            
        return img, boxs, cls_labels, mask
        
class COCODataset(Dataset):
    def __init__(self, data_dir, split, train=False):
        super().__init__()        
        
        self.data_dir = data_dir
        self.split = split
        self.train = train

        self.max_workers=10
        self.verbose=False

        input_shape = [600, 600]
        self.albumentations = Albumentations(input_shape, train) 
        
        ann_file = os.path.join(data_dir, "annotations/instances_{}.json".format(split))
        self.coco = COCO(ann_file)
        self.ids = [str(k) for k in self.coco.imgs if( len(self.coco.loadAnns(self.coco.getAnnIds(k))) > 0)  ]
        # self.ids = [str(k) for k in self.coco.imgs ]

        # ann_ids = self.coco.getAnnIds(img_id)
        # anns = self.coco.loadAnns(ann_ids)
        
        # classes's values must start from 1, because 0 means background in the model
        self.classes = {k: v["name"] for k, v in self.coco.cats.items()}
        
        checked_id_file = os.path.join(data_dir, "checked_{}.txt".format(split))
        # if train:
        #     if not os.path.exists(checked_id_file):
        #         self._aspect_ratios = [v["width"] / v["height"] for v in self.coco.imgs.values()]
        #     self.check_dataset(checked_id_file)

        
    def get_image(self, img_id):
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]   
        image = cv2.imread(os.path.join(self.data_dir, "images/{}".format(self.split), img_info["file_name"])) # h, w, c
        return image
    
    @staticmethod
    def convert_to_xyxy(boxes): # box format: (xmin, ymin, w, h)
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1) # new_box format: (xmin, ymin, xmax, ymax)  

    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann["category_id"])
                mask = self.coco.annToMask(ann)
                masks.append(mask)
         
            masks = np.stack(masks, axis=0)
            masks = np.transpose(masks,(1,2,0))

            boxes = np.stack(boxes, axis=0)
            boxes = np.array(boxes, np.float32)      
  
        return boxes, masks, labels, torch.tensor([img_id])
    
    def check_dataset(self, checked_id_file):
        """
        use multithreads to accelerate the process.
        check the dataset to avoid some problems listed in method `_check`.
        """
        
        if os.path.exists(checked_id_file):
            info = [line.strip().split(", ") for line in open(checked_id_file)]
            self.ids, self.aspect_ratios = zip(*info)
            return
        
        since = time.time()
        print("Checking the dataset...")
        
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        seqs = torch.arange(len(self)).chunk(self.max_workers)
        tasks = [executor.submit(self._check, seq.tolist()) for seq in seqs]

        outs = []
        for future in as_completed(tasks):
            outs.extend(future.result())
        if not hasattr(self, "id_compare_fn"):
            self.id_compare_fn = lambda x: int(x)
        outs.sort(key=lambda x: self.id_compare_fn(x[0]))
        
        with open(checked_id_file, "w") as f:
            for img_id, aspect_ratio in outs:
                f.write("{}, {:.4f}\n".format(img_id, aspect_ratio))
         
        info = [line.strip().split(", ") for line in open(checked_id_file)]
        self.ids, self.aspect_ratios = zip(*info)
        print("checked id file: {}".format(checked_id_file))
        print("{} samples are OK; {:.1f} seconds".format(len(self), time.time() - since))
        
    def _check(self, seq):
        out = []
        for i in seq:
            img_id = self.ids[i]
            target = self.get_target(img_id)
            boxes = target["boxes"]
            labels = target["labels"]
            masks = target["masks"]

            try:
                assert len(boxes) > 0, "{}: len(boxes) = 0".format(i)
                assert len(boxes) == len(labels), "{}: len(boxes) != len(labels)".format(i)
                assert len(boxes) == len(masks), "{}: len(boxes) != len(masks)".format(i)

                out.append((img_id, self._aspect_ratios[i]))
            except AssertionError as e:
                if self.verbose:
                    print(img_id, e)
        return out    

    def __getitem__(self, i):
        img_id = self.ids[i]
        image = self.get_image(img_id)

        # boxes, masks, labels, image_id = self.get_target(img_id) if self.train else {}
        boxes, masks, labels, image_id = self.get_target(img_id) 

        if type(boxes) != list:
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            h, w, c = image.shape   

            boxes = xyxy2xywhn(boxes, w = w, h = h)                  
            image, boxes, labels, masks = self.albumentations(image, boxes, labels, masks)
            boxes = xywhn2xyxy(boxes, w = w, h = h)     

            image = np.transpose(image,(2,0,1)) # 600,600,3 => 3, 600, 600
            image = torch.tensor(image, dtype=torch.float32)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels=torch.tensor(labels, dtype=torch.int64)
            masks=torch.tensor(masks, dtype=torch.uint8)
            target = dict(image_id=image_id, boxes=boxes, labels=labels, masks=masks)  
             # return image, target 
            return image, image_id, boxes, labels, masks
        else:
            print()
       


    def __len__(self):
        return len(self.ids)

# def collate_fn(batch):
#     images = []
#     targets = []
#     for image, target in batch: # only batch 1
#         images.append(image)
#         targets.append(target)

#     images = torch.stack(images)
#     return (images, target)

def collate_fn(batch):
    images = []
    image_ids, boxes, labels, masks = [], [], [], []
    for image, image_id, box, label, mask in batch: # only batch 1
        images.append(image)
        image_ids.append(image_id)
        boxes.append(box)
        labels.append(label)
        masks.append(mask)

    images = torch.stack(images)
    image_ids = torch.stack(image_ids)  

    # boxes = torch.stack(boxes)
    # labels = torch.stack(labels)
    # masks = torch.stack(masks)

    target = dict(image_id=image_ids, boxes=boxes, labels=labels, masks=masks) 
    return (images, target)    
    
    # return (images, image_ids, boxes, labels, masks) 

   