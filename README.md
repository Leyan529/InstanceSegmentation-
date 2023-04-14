# InstanceSegmentation
My Frame work for InstanceSegmentation
## Overview
I organizize the Instance Segmentation algorithms proposed in recent years, and focused on **`Pascal VOC`, `VerSeg vertebra` and `coco` Dataset.**
This frame work also include **`EarlyStopping mechanism`**.


## Datasets:

I used 3 different datases: **`Pascal VOC`, `VerSeg vertebra` and `coco`**. Statistics of datasets I used for experiments is shown below

- **VOC**:
  Download the voc images and annotations from [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007) or [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012). Make sure to put the files as the following structure:

  | Dataset                | Classes | #Train images/objects | #Validation images/objects |
  |------------------------|:---------:|:-----------------------:|:----------------------------:|
  | VOC2007                |    20   |      5011/12608       |           4952/-           |
  | VOC2012                |    20   |      5717/13609       |           5823/13841       |

  -- VOC2007
  * ![](https://i.imgur.com/wncA2wC.png)

  -- VOC2012
  * ![](https://i.imgur.com/v3AQelB.png)

  ```
  VOCDevkit
  ├── VOC2007
  │   ├── Annotations  
  │   ├── ImageSets
  │   ├── JPEGImages
  │   └── ...
  └── VOC2012
      ├── Annotations  
      ├── ImageSets
      ├── JPEGImages
      └── ...
  ```
  Processed File: [download link](https://1drv.ms/u/s!AvbkzP-JBXPAhk51a2a6DLg_Hgub?e=PhUN2s)
  
  
- **VerSeg vertebra**:
The vertebra Dataset is clone from [VerSeg](https://github.com/TWokulski/VerSeg).


  ```
    VERSEG
    ├── Train
    │     ├── annotations.json
    │     ├──  JPEGImages
    │     ├── *.jpg
    │     
    │── Validation
        ├── annotations.json
        ├── JPEGImages
            ├── *.jpg 
  ```
  Processed File: [download link](https://1drv.ms/u/s!AvbkzP-JBXPAhk8nssrL5d6SoOA9?e=uMlmEm)

- **COCO**:
  Download the coco images and annotations from [coco website](http://cocodataset.org/#download). Make sure to put the files as the following structure:

  | Dataset                | Classes | #Train images/objects | #Validation images/objects |
  |------------------------|:---------:|:-----------------------:|:----------------------------:|
  | COCO2014               |    80   |         83k/-         |            41k/-           |
  | COCO2017               |    80   |         118k/-        |             5k/-           |
  ```
    COCO
    ├── annotations
    │   ├── instances_train2014.json
    │   ├── instances_train2017.json
    │   ├── instances_val2014.json
    │   └── instances_val2017.json
    │── images
    │   ├── train2014
    │   ├── train2017
    │   ├── val2014
    │   └── val2017
    └── anno_pickle
        ├── COCO_train2014.pkl
        ├── COCO_val2014.pkl
        ├── COCO_train2017.pkl
        └── COCO_val2017.pkl
  ```
  Processed File: [download link](https://1drv.ms/f/s!AvbkzP-JBXPAhlDiyVy9wsDGCCj8?e=nN58aZ)
  
## Methods
- **MASK_RCNN**
- **Yolact**
#### Pretrain-Weights [download link](https://1drv.ms/f/s!AvbkzP-JBXPAhkCM91SYNDMdKMtL?e=ilEdfK)


## Prerequisites
* **Windows 10**
* **CUDA 10.1 (lower versions may work but were not tested)**
* **NVIDIA GPU 1660 + CuDNN v7.3**
* **python 3.6.9**
* **pytorch 1.10**
* **opencv (cv2)**
* **numpy**
* **torchvision 0.4**

## Requirenents

```python
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage
### 1. Prepare the dataset
* **Create your own `annotation.{dataset}.py` then create `Segmentation/train.txt` , `Segmentation/val.txt` let data to load.** 
* **Prepare pretrain download weight to `model_data` .** 
* **Add new data in `helps/choose_data.py`.**

### 2. Create own model
* **Copy `inst_model` directory and write self required function, like `dataset_collate, Dataset, freeze_backbone, unfreeze_backbone`... etc.** 
* **Maintaion self directory like `nets, utils`.** 
* **Maintaion self detection configuration file like `model.py`.** 
* **Add new data in `helps/choose_model.py`.**

### 3. Train (Freeze backbone + UnFreeze backbone) 
* setup your `root_path` , choose `DataType` and switch segmentation model library import.
```python
python train.py
```

### 4. Evaluate  (get_miou) 
* setup your `root_path` , choose `DataType` and switch detection model library import.
* setup your `model_path` and `classes_path` in `model/model.py`
```python
python eval.py
```

### 5. predict
* Can switch **`predict mode` to detection image** or **`viedo` mode to detection video**
* setup your `model_path` and `classes_path` in `model/model.py`
```python
python predict.py
```

### 6. export
* Can switch your saved model export to ONNX format
```python
python export.py --config "configs.yolact_base"
```
## Demo
![40c](https://user-images.githubusercontent.com/24097516/213660067-c9d14c5d-9aa2-4974-9b12-eb7b2ca99102.png)

## Reference
- https://github.com/bubbliiiing/yolact-pytorch
- https://github.com/TWokulski/VerSeg
- https://github.com/RookieQTZ/mask-rcnn-pytorch
