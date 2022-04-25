## Yolact-keras實例分割模型在pytorch當中的實現
---

## 目錄
1. [性能情況 Performance](#性能情況)
2. [所需環境 Environment](#所需環境)
3. [文件下載 Download](#文件下載)
4. [訓練步驟 How2train](#訓練步驟)
5. [預測步驟 How2predict](#預測步驟)
6. [評估步驟 How2eval](#評估步驟)
7. [參考資料 Reference](#Reference)

## 性能情況
| 訓練數據集 | 權值文件名稱 | 測試數據集 | 輸入圖片大小 | bbox mAP 0.5:0.95 | bbox mAP 0.5 | segm mAP 0.5:0.95 | segm mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: | :-----: | :-----: | 
| COCO-Train2017 | [yolact_weights_coco.pth](https://github.com/bubbliiiing/yolact-pytorch/releases/download/v1.0/yolact_weights_coco.pth) | COCO-Val2017 | 544x544 | 30.4 | 52.0 | 27.3 | 47.7

## 所需環境
pytorch==1.2.0  
torchvision==0.4.0

## 文件下載
訓練所需的預訓練權值可在百度網盤中下載。
鏈接: https://pan.baidu.com/s/1IUlcrGoAJM-ujBV4SsjR8Q     
提取碼: nxjk    

shapes數據集下載地址如下，該數據集是使用labelme標註的結果，尚未經過其它處理，用於區分三角形和正方形：  
鏈接: https://pan.baidu.com/s/1hrCaEYbnSGBOhjoiOKQmig   
提取碼: jk44    

## 訓練步驟
### a、訓練shapes形狀數據集
1. 數據集的準備   
在**文件下載**部分，通過百度網盤下載數據集，下載完成後解壓，將圖片和對應的json文件放入根目錄下的datasets/before文件夾。

2. 數據集的處理   
打開coco_annotation.py，裡面的參數默認用於處理shapes形狀數據集，直接運行可以在datasets/coco文件夾裡生成圖片文件和標籤文件，並且完成了訓練集和測試集的劃分。

3. 開始網絡訓練   
train.py的默認參數用於訓練shapes數據集，默認指向了根目錄下的數據集文件夾，直接運行train.py即可開始訓練。

4. 訓練結果預測   
訓練結果預測需要用到兩個文件，分別是yolact.py和predict.py。
首先需要去yolact.py裡面修改model_path以及classes_path，這兩個參數必須要修改。
**model_path指向訓練好的權值文件，在logs文件夾裡。
classes_path指向檢測類別所對應的txt。**    
完成修改後就可以運行predict.py進行檢測了。運行後輸入圖片路徑即可檢測。

### b、訓練自己的數據集
1. 數據集的準備  
**本文使用labelme工具進行標註，標註好的文件有圖片文件和json文件，二者均放在before文件夾裡，具體格式可參考shapes數據集。**    
在標註目標時需要注意，同一種類的不同目標需要使用 _ 來隔開。
比如想要訓練網絡檢測**三角形和正方形**，當一幅圖片存在兩個三角形時，分別標記為：   
```python
triangle_1
triangle_2
```
2. 數據集的處理  
修改coco_annotation.py裡面的參數。第一次訓練可以僅修改classes_path，classes_path用於指向檢測類別所對應的txt。
訓練自己的數據集時，可以自己建立一個cls_classes.txt，裡面寫自己所需要區分的類別。
model_data/cls_classes.txt文件內容為：      
```python
cat
dog
...
```  
修改coco_annotation.py中的classes_path，使其對應cls_classes.txt，並運行coco_annotation.py。

3. 開始網絡訓練  
**訓練的參數較多，均在train.py中，大家可以在下載庫後仔細看註釋，其中最重要的部分依然是train.py裡的classes_path。**   
**classes_path用於指向檢測類別所對應的txt，這個txt和coco_annotation.py裡面的txt一樣！訓練自己的數據集必須要修改！**    
修改完classes_path後就可以運行train.py開始訓練了，在訓練多個epoch後，權值會生成在logs文件夾中。

4. 訓練結果預測  
訓練結果預測需要用到兩個文件，分別是yolact.py和predict.py。
首先需要去yolact.py裡面修改model_path以及classes_path，這兩個參數必須要修改。
**model_path指向訓練好的權值文件，在logs文件夾裡。
classes_path指向檢測類別所對應的txt。**     
完成修改後就可以運行predict.py進行檢測了。運行後輸入圖片路徑即可檢測。

### c、訓練coco數據集
1. 數據集的準備  
coco訓練集 http://images.cocodataset.org/zips/train2017.zip   
coco驗證集 http://images.cocodataset.org/zips/val2017.zip   
coco訓練集和驗證集的標籤 http://images.cocodataset.org/annotations/annotations_trainval2017.zip   

2. 開始網絡訓練  
解壓訓練集、驗證集及其標籤後。打開train.py文件，修改其中的classes_path指向model_data/coco_classes.txt。
修改train_image_path為訓練圖片的路徑，train_annotation_path為訓練圖片的標籤文件，val_image_path為驗證圖片的路徑，val_annotation_path為驗證圖片的標籤文件。

3. 訓練結果預測  
訓練結果預測需要用到兩個文件，分別是yolact.py和predict.py。
首先需要去yolact.py裡面修改model_path以及classes_path，這兩個參數必須要修改。
**model_path指向訓練好的權值文件，在logs文件夾裡。
classes_path指向檢測類別所對應的txt。**    
完成修改後就可以運行predict.py進行檢測了。運行後輸入圖片路徑即可檢測。

## 預測步驟
### a、使用預訓練權重
1. 下載完庫後解壓，在百度網盤下載權值，放入model_data，運行predict.py，輸入   
```python
img/street.jpg
```
2. 在predict.py裡面進行設置可以進行fps測試和video視頻檢測。
### b、使用自己訓練的權重
1. 按照訓練步驟訓練。
2. 在yolact.py文件裡面，在如下部分修改model_path和classes_path使其對應訓練好的文件；**model_path對應logs文件夾下面的權值文件，classes_path是model_path對應分的類**。
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己訓練好的模型進行預測一定要修改model_path和classes_path！
    #   model_path指向logs文件夾下的權值文件，classes_path指向model_data下的txt
    #
    #   訓練好後logs文件夾下存在多個權值文件，選擇驗證集損失較低的即可。
    #   驗證集損失較低不代表mAP較高，僅代表該權值在驗證集上泛化性能較好。
    #   如果出現shape不匹配，同時要注意訓練時的model_path和classes_path參數的修改
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/yolact_weights_shape.pth',
    "classes_path"      : 'model_data/shape_classes.txt',
    #---------------------------------------------------------------------#
    #   輸入圖片的大小
    #---------------------------------------------------------------------#
    "input_shape"       : [544, 544],
    #---------------------------------------------------------------------#
    #   只有得分大於置信度的預測框會被保留下來
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   非極大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #---------------------------------------------------------------------#
    #   先驗框的大小
    #---------------------------------------------------------------------#
    "anchors_size"      : [24, 48, 96, 192, 384],
    #---------------------------------------------------------------------#
    #   傳統非極大抑制
    #---------------------------------------------------------------------#
    "traditional_nms"   : True,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True
}
```
3. 運行predict.py，輸入    
```python
img/street.jpg
```
4. 在predict.py裡面進行設置可以進行fps測試和video視頻檢測。

## 評估步驟 
### a、評估自己的數據集
1. 本文使用coco格式進行評估。
2. 如果在訓練前已經運行過coco_annotation.py文件，代碼會自動將數據集劃分成訓練集、驗證集和測試集。
3. 如果想要修改測試集的比例，可以修改coco_annotation.py文件下的trainval_percent。 trainval_percent用於指定(訓練集+驗證集)與測試集的比例，默認情況下 (訓練集+驗證集):測試集 = 9:1。 train_percent用於指定(訓練集+驗證集)中訓練集與驗證集的比例，默認情況下 訓練集:驗證集 = 9:1。
4. 在yolact.py裡面修改model_path以及classes_path。 **model_path指向訓練好的權值文件，在logs文件夾裡。 classes_path指向檢測類別所對應的txt。**    
5. 前往eval.py文件修改classes_path，classes_path用於指向檢測類別所對應的txt，這個txt和訓練時的txt一樣。評估自己的數據集必須要修改。運行eval.py即可獲得評估結果。

### b、評估coco的數據集
1. 下載好coco數據集。
2. 在yolact.py裡面修改model_path以及classes_path。 **model_path指向coco數據集的權重，在logs文件夾裡。 classes_path指向model_data/coco_classes.txt。**    
3. 前往eval.py設置classes_path，指向model_data/coco_classes.txt。修改Image_dir為評估圖片的路徑，Json_path為評估圖片的標籤文件。運行eval.py即可獲得評估結果。
  
## Reference
1. https://github.com/bubbliiiing/yolact-pytorch
2. https://github.com/feiyuhuahuo/Yolact_minimal   
3. https://github.com/PanJinquan/python-learning-notes