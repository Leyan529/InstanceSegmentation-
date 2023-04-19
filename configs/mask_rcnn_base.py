import sys
sys.path.append(".")

import argparse, os, json
import torch
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter
from utils.tools import init_logging
import importlib
from utils.helpers import get_classes, get_anchors, get_data


def get_opts(Train=True):
    opt = argparse.Namespace()  

    #the train data, you need change.
    opt.data_root = '/home/leyan/DataSet/'
    # opt.data_root = "/home/zimdytsai/leyan/DataSet"
    # opt.data_root = 'D://WorkSpace//JupyterWorkSpace//DataSet//'


    opt.out_root = 'work_dirs/'
    opt.exp_name = 'coco'
    """
    [ voc, verseg, coco ]
    """
    # get annotation file in current seting    

    opt.data_path, opt.classes_path = get_data(opt.data_root, opt.exp_name)
    # importlib.import_module("annotation.{}".format(opt.exp_name)).get_annotation(opt.data_root, opt.classes_path) 
    #############################################################################################
    #############################################################################################    
    opt.net = 'Mask_RCNN'     # [yolact, Mask_RCNN]
    opt.model_path      = '' 
    opt.input_shape     = None  
    opt.pretrained      = True
    opt.IM_SHAPE = (544, 544, 3)
    #------------------------------------------------------#
    #   统计所有图像比例在bins区间中的位置索引
    #------------------------------------------------------#
    opt.aspect_ratio_group_factor    = 3
    #---------------------------------------------------------#
    #   下采樣的倍數8、16 
    #   8下采樣的倍數較小、理論上效果更好，但也要求更大的顯存
    #---------------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    opt.train_image_path, opt.val_image_path, opt.train_coco, opt.val_coco, opt.class_names, opt.num_classes, opt.COCO_LABEL_MAP = \
     importlib.import_module("annotation.{}".format(opt.exp_name)).get_annotation(
        opt.data_root, opt.classes_path) 

    opt.num_classes = opt.num_classes + 1
    #---------------------------#
    #   讀取數據集對應的txt
    #---------------------------# 
    opt.num_train   = len(list(opt.train_coco.imgToAnns.keys()))
    opt.num_val     = len(list(opt.val_coco.imgToAnns.keys())) 
    #------------------------------------------------------------------#
    opt.Cosine_lr           = False
    opt.label_smoothing     = 0
    #----------------------------------------------------#
    #   凍結階段訓練參數
    #   此時模型的主幹被凍結了，特征提取網絡不發生改變
    #   占用的顯存較小，僅對網絡進行微調
    #----------------------------------------------------#
    opt.ngpu = 2
    opt.Init_Epoch          = 0
    opt.Freeze_Epoch    = 50 #50
    opt.Freeze_batch_size   = int(4/2)
    opt.Freeze_lr           = 1e-3
    #----------------------------------------------------#
    #   解凍階段訓練參數
    #   此時模型的主幹不被凍結了，特征提取網絡會發生改變
    #   占用的顯存較大，網絡所有的參數都會發生改變
    #----------------------------------------------------#
    opt.UnFreeze_Epoch  = 100 #100
    opt.Unfreeze_batch_size = int(2/1)
    opt.Unfreeze_lr         = 1e-4
    #------------------------------------------------------#
    #   是否進行凍結訓練，默認先凍結主幹訓練後解凍訓練。
    #------------------------------------------------------#
    opt.Freeze_Train        = True
    #---------------------------------------------------------------------# 
    #   建議選項：
    #   種類少（幾類）時，設置為True
    #   種類多（十幾類）時，如果batch_size比較大（10以上），那麼設置為True
    #   種類多（十幾類）時，如果batch_size比較小（10以下），那麼設置為False
    #---------------------------------------------------------------------# 
    opt.dice_loss       = False
    #---------------------------------------------------------------------# 
    #   是否使用focal loss來防止正負樣本不平衡
    #---------------------------------------------------------------------# 
    opt.focal_loss      = True
    #-------------------------------------------------------------------#
    #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
    #-------------------------------------------------------------------#
    opt.batch_size = opt.Freeze_batch_size if opt.Freeze_Train else opt.Unfreeze_batch_size
    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    opt.Init_lr             = 1e-2
    opt.Min_lr              = opt.Init_lr * 0.01
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #------------------------------------------------------------------#
    opt.lr_decay_type       = "cos"
    opt.weight_decay    = 5e-4
    opt.gamma           = 0.94
    opt.optimizer_type      = "sgd"
    opt.momentum            = 0.937
    #------------------------------------------------------#
    #   是否提早結束。
    #------------------------------------------------------#
    opt.Early_Stopping  = True
    #------------------------------------------------------#
    #   主幹特征提取網絡特征通用，凍結訓練可以加快訓練速度
    #   也可以在訓練初期防止權值被破壞。
    #   Init_Epoch為起始世代
    #   Freeze_Epoch為凍結訓練的世代
    #   UnFreeze_Epoch總訓練世代
    #   提示OOM或者顯存不足請調小Batch_size
    #------------------------------------------------------#
    opt.UnFreeze_flag = False
    #-------------------------------------------------------------------#
    #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
    #-------------------------------------------------------------------#
    opt.batch_size = opt.Freeze_batch_size if opt.Freeze_Train else opt.Unfreeze_batch_size
    opt.end_epoch = opt.Freeze_Epoch if opt.Freeze_Train else opt.UnFreeze_Epoch
    #------------------------------------------------------#
    #   用於設置是否使用多線程讀取數據
    #   開啟後會加快數據讀取速度，但是會占用更多內存
    #   內存較小的電腦可以設置為2或者0  
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #------------------------------------------------------#
    opt.num_workers         = 1
    opt.Cuda                = True
    opt.distributed         = True
    opt.sync_bn             = True
    opt.fp16                = True
    #############################################################################################
    opt.debug = 0
    ### Other ###
    opt.manual_seed = 704
    opt.log_batch_interval = 10
    opt.log_checkpoint = 10
    try:
        opt.local_rank  = int(os.environ["LOCAL_RANK"])
    except:
        opt.local_rank  = 0
    opt.ngpus_per_node  = torch.cuda.device_count()
    #############################################################################################
    opt.out_path = os.path.join(opt.out_root, "{}_{}".format(opt.exp_name, opt.net))
    if Train:
        opt.writer = SummaryWriter(log_dir=os.path.join(opt.out_path, "tensorboard"))
        init_logging(opt.local_rank, opt.out_path)    
 
    else:
        from inst_model.Mask_RCNN.mask_rcnn import Mask_RCNN
        opt.Model_Pred = Mask_RCNN

        if opt.exp_name == 'coco':
            year = 2014
            opt.Image_dir = os.path.join(opt.data_root, 
                    f"COCO/val{year}")
            opt.Json_path = os.path.join(opt.data_root, 
                    f"COCO/annotations_trainval{year}/annotations/instances_val{year}.json")

        elif opt.exp_name == 'voc':
            year = 2012
            opt.Image_dir = os.path.join(opt.data_root, 
                    f"VOCdevkit/VOC{year}/JPEGImages")
            opt.Json_path = os.path.join(opt.data_root, 
                    f"VOCdevkit/VOC{year}/Annotations/VOC{year}.json")

 
    return opt

if __name__ == "__main__":    
    get_opts(Train=False)


