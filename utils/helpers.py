import numpy as np
import os
import math
import torch
import torch.nn as nn
from copy import deepcopy

from pynvml import *
from functools import partial
import threading
from itertools import product
from math import sqrt
from pycocotools.coco import COCO

def get_data(root_path, dataType):
    #------------------------------#  
    #   數據集路徑
    #   訓練自己的數據集必須要修改的
    #------------------------------#    
    if dataType == "voc":
        data_path = os.path.join(root_path, 'VOCdevkit')    
        classes_path    = 'model_data/voc_classes.txt'   

    return data_path, classes_path


class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
    
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)    

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def get_data_path(root_path, dataType):
    #------------------------------#  
    #   數據集路徑
    #   訓練自己的數據集必須要修改的
    #------------------------------#  
    map_dict = { "AsianTraffic":'Asian-Traffic', "bdd":'bdd100k', "coco":'COCO',
                 "voc":'VOCdevkit', "lane":"LANEdevkit", "widerperson":'WiderPerson', 
                 "MosquitoContainer":'MosquitoContainer' }
    return os.path.join(root_path, map_dict[dataType]) 

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)    

def make_anchors(conv_h, conv_w, scale, input_shape=[550, 550], aspect_ratios=[1, 1 / 2, 2]):
    prior_data = []
    for j, i in product(range(conv_h), range(conv_w)):
        x = (i + 0.5) / conv_w
        y = (j + 0.5) / conv_h

        for ar in aspect_ratios:
            ar = sqrt(ar)
            w = scale * ar / input_shape[1]
            h = scale / ar / input_shape[0]

            prior_data += [x, y, w, h]

    return prior_data

#---------------------------------------------------#
#   用于计算共享特征层的大小
#---------------------------------------------------#
def get_img_output_length(height, width):
    filter_sizes    = [7, 3, 3, 3, 3, 3, 3]
    padding         = [3, 1, 1, 1, 1, 1, 1]
    stride          = [2, 2, 2, 2, 2, 2, 2]
    feature_heights = []
    feature_widths  = []

    for i in range(len(filter_sizes)):
        height  = (height + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        width   = (width + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    return np.array(feature_heights)[-5:], np.array(feature_widths)[-5:]
    
def get_anchors(input_shape = [550, 550], anchors_size = [24, 48, 96, 192, 384]):
    feature_heights, feature_widths = get_img_output_length(input_shape[0], input_shape[1])
    
    all_anchors = []
    for i in range(len(feature_heights)):
        anchors     = make_anchors(feature_heights[i], feature_widths[i], anchors_size[i], input_shape)
        all_anchors += anchors
    
    all_anchors = np.reshape(all_anchors, [-1, 4])
    return all_anchors   

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

class LossHistory():
    def __init__(self, opt, patience = 10):        
        self.losses     = []
        self.val_loss   = []
        self.writer = opt.writer
        self.freeze = False
        self.log_dir = opt.out_path
       
        if opt.local_rank == 0:
            # launch tensorboard
            t = threading.Thread(target=self.launchTensorBoard, args=([opt.out_path]))
            t.start()       

        # initial EarlyStopping
        self.patience = patience
        self.reset_stop()          

    def launchTensorBoard(self, tensorBoardPath, port = 8888):
        os.system('tensorboard --logdir=%s --port=%s --load_fast=false'%(tensorBoardPath, port))
        url = "http://localhost:%s/"%(port)
        # webbrowser.open_new(url)
        return

    def reset_stop(self):
        self.best_epoch_loss = np.Inf 
        self.stopping = False
        self.counter  = 0

    def set_status(self, freeze):
        self.freeze = freeze

    def epoch_loss(self, loss, val_loss, epoch):
        self.losses.append(loss)
        self.val_loss.append(val_loss)  

        prefix = "Freeze_epoch/" if self.freeze else "UnFreeze_epoch/"     
        self.writer.add_scalar(prefix+'Loss/Train', loss, epoch)
        self.writer.add_scalar(prefix+'Loss/Val', val_loss, epoch)
        self.decide(val_loss)   

    def step(self, steploss, iteration):        
        prefix = "Freeze_step/" if self.freeze else "UnFreeze_step/"
        self.writer.add_scalar(prefix + 'Train/Loss', steploss, iteration)
         

    def decide(self, epoch_loss):
        if epoch_loss > self.best_epoch_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print(f'Best lower loss:{self.best_epoch_loss}')
                self.stopping = True
        else:
            self.best_epoch_loss = epoch_loss           
            self.counter = 0 
            self.stopping = False