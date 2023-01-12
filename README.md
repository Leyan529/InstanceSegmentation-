# InstanceSegmentation
My Frame work for InstanceSegmentation
## Overview
I organizize the Instance Segmentation algorithms proposed in recent years, and focused on **`Pascal VOC`, `VerSeg vertebra` Dataset.**
This frame work also include **`EarlyStopping mechanism`**.


## Datasets:

I used 2 different datases: **`Pascal VOC`, `VerSeg vertebra`. Statistics of datasets I used for experiments is shown below

- **VOC**:
  Download the voc images and annotations from [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007) or [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012). Make sure to put the files as the following structure:

-- VOC2007
![](https://i.imgur.com/wncA2wC.png)

-- VOC2012
![](https://i.imgur.com/v3AQelB.png)

  
  
| Dataset                | Classes | #Train images/objects | #Validation images/objects |
|------------------------|:---------:|:-----------------------:|:----------------------------:|
| VOC2007                |    20   |      209/633       |          213/582        |
| VOC2012                |    20   |      1464/3507     |         1449/3422       |

  ```
  VOCDevkit
  ├── VOC2007
  │   ├── JPEGImages  
  │   ├── SegmentationClass
  │   ├── ...
  │   └── ...
  └── VOC2012
      ├── JPEGImages  
      ├── SegmentationClass
      ├── ...
      └── ...
  ```
  
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



## Methods
- **MASK_RCNN**
- **Yolact**


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
* **Add new data in `helps/choose_data.py`. **

### 2. Create own model
* **Copy `inst_model` directory and write self required function, like `dataset_collate, Dataset, freeze_backbone, unfreeze_backbone`... etc.** 
* **Maintaion self directory like `nets, utils`. ** 
* **Maintaion self detection configuration file like `model.py`. ** 
* **Add new data in `helps/choose_model.py`. **

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

## Reference
- https://github.com/bubbliiiing/yolact-pytorch
- https://github.com/TWokulski/VerSeg
