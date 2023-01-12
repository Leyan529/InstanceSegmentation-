import importlib
import threading
from tqdm import tqdm
import torch
from utils.helpers import get_lr
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import os


def fit_yolact(model_train, model, criterion, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, opt):
    Epoch, cuda, fp16, scaler, ema, local_rank = opt.end_epoch, opt.Cuda, opt.fp16, opt.scaler, opt.ema, opt.local_rank
    
    total_loss  = 0
    val_loss    = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets, masks_gt, num_crowds = batch[0], batch[1], batch[2], batch[3]
        with torch.no_grad():
            if cuda:
                images      = images.cuda(local_rank)
                targets     = [ann.cuda(local_rank) for ann in targets]
                masks_gt    = [mask.cuda(local_rank) for mask in masks_gt]
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(images)
            #----------------------#
            #   计算损失
            #----------------------#
            losses  = criterion(outputs, targets, masks_gt, num_crowds)
            losses  = {k: v.mean() for k, v in losses.items()}  # Mean here because Dataparallel.
            loss    = sum([losses[k] for k in losses])

            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(images)
                #----------------------#
                #   计算损失
                #----------------------#
                losses  = criterion(outputs, targets, masks_gt, num_crowds)
                losses  = {k: v.mean() for k, v in losses.items()}  # Mean here because Dataparallel.
                loss    = sum([losses[k] for k in losses])

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        total_loss += loss.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
        loss_history.step(total_loss / (iteration + 1), (epoch_step * epoch + iteration + 1))

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    # model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets, masks_gt, num_crowds = batch[0], batch[1], batch[2], batch[3]
        with torch.no_grad():
            if cuda:
                images      = images.cuda(local_rank)
                targets     = [ann.cuda(local_rank) for ann in targets]
                masks_gt    = [mask.cuda(local_rank) for mask in masks_gt]

            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train_eval(images)
            #----------------------#
            #   计算损失
            #----------------------#
            losses      = criterion(outputs, targets, masks_gt, num_crowds)
            losses      = {k: v.mean() for k, v in losses.items()}
            loss        = sum([losses[k] for k in losses])

            val_loss += loss.item()
            
            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1), 
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.epoch_loss(total_loss / epoch_step, val_loss / epoch_step_val, epoch+1)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        torch.save(save_state_dict, '%s/ep%03d-loss%.3f-val_loss%.3f.pth' % (loss_history.log_dir, epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val))
        # best epoch weights
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "best_epoch_weights.pth"))
        # last epoch weights
        torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "last_epoch_weights.pth"))


def fit_mask_rcnn(model_train, model, criterion, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, opt):
    Epoch, cuda, fp16, scaler, ema, local_rank = opt.end_epoch, opt.Cuda, opt.fp16, opt.scaler, opt.ema, opt.local_rank
    
    total_loss  = 0
    val_loss    = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        batch_loss = []
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images      = [img.cuda(local_rank) for img in images]
                targets     = [ {key:  target[key].cuda(local_rank) for key in target.keys()}
                                for target in targets                                                            
                ]
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            for image, target in zip(images, targets):
                #----------------------#
                #   前向传播 + 计算损失
                #----------------------#
                losses = model_train(image, target)
                total_loss = sum([losses[k] for k in losses])
                batch_loss.append(total_loss)
            loss = torch.stack(batch_loss, dim=0).sum(dim=0).sum(dim=0) / len(batch_loss)   

            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                for image, target in zip(images, targets):
                    #----------------------#
                    #   前向传播 + 计算损失
                    #----------------------#
                    losses = model_train(image, target)
                    total_loss = sum([losses[k] for k in losses])
                    batch_loss.append(total_loss)
                loss = torch.stack(batch_loss, dim=0).sum(dim=0).sum(dim=0) / len(batch_loss)                
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        total_loss += loss.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
        loss_history.step(total_loss.item() / (iteration + 1), (epoch_step * epoch + iteration + 1))

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        batch_loss = []
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images      = [img.cuda(local_rank) for img in images]
                targets     = [ {key:  target[key].cuda(local_rank) for key in target.keys()}
                                for target in targets                                                            
                ]
            optimizer.zero_grad()
            for image, target in zip(images, targets):
                #----------------------#
                #   前向传播 + 计算损失
                #----------------------#
                losses = model_train_eval(image, target)
                total_loss = sum([losses[k] for k in losses])
                batch_loss.append(total_loss)                
            loss = torch.stack(batch_loss, dim=0).sum(dim=0).sum(dim=0) / len(batch_loss) 
            val_loss += loss.item()
            
            if local_rank == 0:
                pbar.set_postfix(**{
                                    'val_loss': val_loss / (iteration + 1), 
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.epoch_loss(total_loss / epoch_step, val_loss / epoch_step_val, epoch+1)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        # print('Total Loss: %.3f ' % (total_loss / epoch_step))        
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))   
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        torch.save(save_state_dict, '%s/ep%03d-loss%.3f-val_loss%.3f.pth' % (loss_history.log_dir, epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val))
        # best epoch weights
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "best_epoch_weights.pth"))
        # last epoch weights
        torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "last_epoch_weights.pth"))


def get_fit_func(opt):
    if opt.net == 'yolact':
        return fit_yolact  
    elif opt.net == 'Mask_RCNN':
        return fit_mask_rcnn
