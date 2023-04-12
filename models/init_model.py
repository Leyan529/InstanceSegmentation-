import torch
import torch.nn as nn

import importlib
import torch.optim as optim
from models.transform import Augmentation
from annotation.train_utils.coco_utils import CocoDetection

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
        val_dataset = CocoDetection(opt.val_image_path, opt.val_coco, dataset="val", net_type = opt.net, label_map = opt.COCO_LABEL_MAP, augmentation=Augmentation(opt.input_shape))
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
            val_dataset     = yolactDataset(opt.val_image_path, opt.val_coco, opt.COCO_LABEL_MAP, Augmentation(opt.input_shape))
            dataset_collate = yolact_dataset_collate        

        elif opt.net == 'Mask_RCNN':
            from inst_model.Mask_RCNN.utils.dataloader import MaskDataset, mask_dataset_collate
            # data_transform = {
            #     "train": transforms.Compose([transforms.ToTensor(),
            #                                 transforms.RandomHorizontalFlip(0.5)]),
            #     "val": transforms.Compose([transforms.ToTensor()])
            # }        
            train_dataset   = MaskDataset(opt.train_image_path, opt.train_coco, opt.COCO_LABEL_MAP, Augmentation(opt.input_shape))
            val_dataset     = MaskDataset(opt.val_image_path, opt.val_coco, opt.COCO_LABEL_MAP, Augmentation(opt.input_shape))
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

    gen             = torch.utils.data.DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
    gen_val         = torch.utils.data.DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=dataset_collate, sampler=val_sampler) 
    return gen, gen_val