import bisect
import glob
import os
import re
import time

import torch

from mask_rcnn.utils.utils_fit import train_one_epoch, eval_one_epoch
from mask_rcnn.net import maskrcnn_resnet50
from mask_rcnn.utils.utils import get_gpu_prop, save_ckpt, collect_gpu_info

from mask_rcnn.utils.dataloader import COCODataset, collate_fn

#    
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda": 
        get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
        
    # ---------------------- prepare data loader ------------------------------- #
    dataset_train = COCODataset( args.data_dir, "train2017", train=True)    
    d_test = COCODataset(args.data_dir, "val2017", train=True) # set train=True for eval  

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=4, shuffle=True, num_workers=1, drop_last = True,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        d_test, batch_size=2, shuffle=False, num_workers=1, drop_last = True,
        collate_fn=collate_fn)  
        
    args.warmup_iters = max(1000, len(data_loader))
    
    # -------------------------------------------------------------------------- #

    print(args)
    num_classes = max(dataset_train.classes.keys()) + 1 # including background class
    model = maskrcnn_resnet50(True, num_classes).to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_lambda = lambda x: 0.1 ** bisect.bisect(args.lr_steps, x)
    
    start_epoch = 0
    
    # find all checkpoints, and load the latest checkpoint
    prefix, ext = os.path.splitext(args.ckpt_path)
    # ckpts = glob.glob(prefix + "-*" + ext)
    ckpts = glob.glob(prefix + "*" + ext)
    # ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device) # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]
        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))
    
    # ------------------------------- train ------------------------------------ #
        
    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))
            
        A = time.time()
        args.lr_epoch = lr_lambda(epoch) * args.lr
        print("lr_epoch: {:.5f}, factor: {:.5f}".format(args.lr_epoch, lr_lambda(epoch)))
        loss, iter_train = train_one_epoch(model, optimizer, data_loader, device, epoch, args)
        A = time.time() - A
        
        B = time.time()
        loss_eval, iter_eval = eval_one_epoch(model, data_loader_test, device, epoch, args)
        # eval_output, iter_eval = pmr.evaluate(model, d_test, device, args)
        B = time.time() - B
        # torch.save(model.state_dict(), '%s/ep%03d-loss%.3f-val_loss%.3f.pth' % ("logs", epoch + 1, iter_train, iter_eval))

        trained_epoch = epoch + 1
        print("------------------------------------------------------------")
        print("training: {:.1f} s, evaluation: {:.1f} s".format(A, B))
        collect_gpu_info("maskrcnn", [1 / iter_train, 1 / iter_eval])
        # print(eval_output.get_AP())

        path = '%s/maskrcnn_ep%03d-loss%.3f-val_loss%.3f.pth' % ("logs", epoch + 1, loss, loss_eval)
        save_ckpt(model, optimizer, trained_epoch, path)

        # it will create many checkpoint files during training, so delete some.
        prefix, ext = os.path.splitext(args.ckpt_path)
        # ckpts = glob.glob(prefix + "-*" + ext)
        ckpts = glob.glob(prefix + "*" + ext)
        # ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        n = 5
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.remove(ckpts[i])
                print("remove {}".format(ckpts[i]))
        
    # -------------------------------------------------------------------------- #

    print("\ntotal time of this training: {:.1f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true", default=True)
    
    parser.add_argument("--dataset", default="coco", help="coco or voc")
    parser.add_argument("--data-dir", default="D:\WorkSpace\JupyterWorkSpace\DataSet\COCO")
    parser.add_argument("--ckpt-path", default="logs/maskrcnn_ep.pth")
    parser.add_argument("--results")
    
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[6, 7])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iters", type=int, default=10, help="max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")
    args = parser.parse_args()
    
    if args.lr is None:
        args.lr = 0.02 * 1 / 16 # lr should be 'batch_size / 16 * 0.02'
    if args.ckpt_path is None:
        args.ckpt_path = "./maskrcnn_{}.pth".format(args.dataset)
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.pth")
    
    main(args)
    
    