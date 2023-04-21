import torch
import torch.nn as nn
import os
import importlib
import torch.optim as optim
from models.transform import Augmentation, BaseTransform
from annotation.train_utils.coco_utils import CocoDetection
from utils.utils import create_aspect_ratio_groups, GroupedBatchSampler

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    # print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_model(opt, train_mode=True):  
    model = init_dt_model(opt, train_mode)
    criterion = init_loss(opt)   
    return model, criterion

def init_dt_model(opt, train_mode=True):
    if opt.net == 'yolact':
        from inst_model.yolact.nets.yolact import Yolact
        model = Yolact(num_classes=opt.num_classes, pretrained=opt.pretrained, train_mode=train_mode)
    elif opt.net == 'Mask_RCNN':       
        from inst_model.Mask_RCNN.net.backbone import resnet50_fpn_backbone
        from inst_model.Mask_RCNN.net.network_files import MaskRCNN
        backbone = resnet50_fpn_backbone(pretrain_path="model_data/weight/resnet50.pth", trainable_layers=3)
        model = MaskRCNN(backbone, num_classes=opt.num_classes, use_pre_trained=opt.pretrained, train_mode=train_mode)

    if not train_mode: return model.eval()
    return model      

def init_loss(opt):
    if opt.net == 'yolact':
        from inst_model.yolact.nets.yolact_training import Multi_Loss
        criterion       = Multi_Loss(opt.num_classes, opt.anchors, 0.5, 0.4, 3)  
    else:
        criterion = None
    return criterion 

def get_optimizer(model, opt, optimizer_type):    
    optimizer = {
            'adam'  : optim.Adam(model.parameters(), opt.Init_lr_fit, betas = (opt.momentum, 0.999), weight_decay = opt.weight_decay),
            'adamw' : optim.AdamW(model.parameters(), opt.Init_lr_fit, betas = (opt.momentum, 0.999), weight_decay = opt.weight_decay),
            'sgd'   : optim.SGD(model.parameters(), opt.Init_lr_fit, momentum = opt.momentum, nesterov=True, weight_decay = opt.weight_decay)
        }[optimizer_type]   
    return optimizer

def generate_loader(opt):
    if opt.exp_name == "coco":
        train_dataset = CocoDetection(opt.train_image_path, opt.train_coco, dataset="train", net_type = opt.net, label_map = opt.COCO_LABEL_MAP, augmentation=Augmentation(opt.input_shape))
        val_dataset = CocoDetection(opt.val_image_path, opt.val_coco, dataset="val", net_type = opt.net, label_map = opt.COCO_LABEL_MAP, augmentation=BaseTransform(opt.input_shape))
        if opt.net == 'yolact':
            from inst_model.yolact.utils.dataloader import yolact_dataset_collate        
            dataset_collate = yolact_dataset_collate 
        else:
            from inst_model.Mask_RCNN.utils.dataloader import mask_dataset_collate 
            dataset_collate = mask_dataset_collate 
    else:
        if opt.net == 'yolact':
            from inst_model.yolact.utils.dataloader import yolactDataset, yolact_dataset_collate        
            train_dataset   = yolactDataset(opt.train_image_path, opt.train_coco, opt.COCO_LABEL_MAP, Augmentation(opt.input_shape))
            val_dataset     = yolactDataset(opt.val_image_path, opt.val_coco, opt.COCO_LABEL_MAP, BaseTransform(opt.input_shape))
            dataset_collate = yolact_dataset_collate        

        elif opt.net == 'Mask_RCNN':
            from inst_model.Mask_RCNN.utils.dataloader import MaskDataset, mask_dataset_collate       
            train_dataset   = MaskDataset(opt.train_image_path, opt.train_coco, opt.COCO_LABEL_MAP, Augmentation(opt.input_shape))
            val_dataset     = MaskDataset(opt.val_image_path, opt.val_coco, opt.COCO_LABEL_MAP, BaseTransform(opt.input_shape))           
            dataset_collate = mask_dataset_collate     

    batch_size      = opt.batch_size
    if opt.distributed:
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
        batch_size      = batch_size // opt.ngpus_per_node
        shuffle         = False
    else:
        train_sampler   = None
        val_sampler     = None
        shuffle         = True

    if opt.net == 'Mask_RCNN':
        # 是否按图片相似高宽比采样图片组成batch
        # 使用的话能够减小训练时所需GPU显存，默认使用
        if opt.aspect_ratio_group_factor >= 0:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            # 统计所有图像高宽比例在bins区间中的位置索引
            group_ids = create_aspect_ratio_groups(train_dataset, k=opt.aspect_ratio_group_factor)
            # 每个batch图片从同一高宽比例区间中取
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, opt.batch_size)
        
        # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
        batch_size = opt.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using %g dataloader workers' % nw)

        if train_sampler:
            # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
            gen = torch.utils.data.DataLoader(train_dataset,
                                                batch_sampler=train_batch_sampler,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=dataset_collate)
            gen_val = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=dataset_collate)
        else:
            gen             = torch.utils.data.DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=dataset_collate, sampler=None)
            gen_val         = torch.utils.data.DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=dataset_collate, sampler=val_sampler) 
    
    else:
        gen             = torch.utils.data.DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
        gen_val         = torch.utils.data.DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=dataset_collate, sampler=val_sampler) 
    return gen, gen_val