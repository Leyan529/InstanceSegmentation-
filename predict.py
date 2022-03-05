import bisect
import glob
import os
import re
import time

import torch

from mask_rcnn.utils.utils_fit import train_one_epoch, eval_one_epoch
from mask_rcnn.net import maskrcnn_resnet50
from mask_rcnn.utils.utils import get_gpu_prop, save_ckpt

import torch.nn.functional as F
import cv2
import numpy as np

import albumentations as argu

def expand_detection(mask, box, padding):
    M = mask.shape[-1]
    scale = (M + 2 * padding) / M
    padded_mask = torch.nn.functional.pad(mask, (padding,) * 4)
    
    w_half = (box[:, 2] - box[:, 0]) * 0.5
    h_half = (box[:, 3] - box[:, 1]) * 0.5
    x_c = (box[:, 2] + box[:, 0]) * 0.5
    y_c = (box[:, 3] + box[:, 1]) * 0.5

    w_half = w_half * scale
    h_half = h_half * scale

    box_exp = torch.zeros_like(box)
    box_exp[:, 0] = x_c - w_half
    box_exp[:, 2] = x_c + w_half
    box_exp[:, 1] = y_c - h_half
    box_exp[:, 3] = y_c + h_half
    return padded_mask, box_exp.to(torch.int64)

@torch.no_grad()
def paste_masks_in_image(mask, box, padding, image_shape):
    mask, box = expand_detection(mask, box, padding)
    
    N = mask.shape[0]
    size = (N,) + tuple(image_shape)
    im_mask = torch.zeros(size, dtype=mask.dtype, device=mask.device)
    for m, b, im in zip(mask, box, im_mask):
        b = b.tolist()
        w = max(b[2] - b[0], 1)
        h = max(b[3] - b[1], 1)
        
        m = F.interpolate(m[None, None], size=(h, w), mode='bilinear', align_corners=False)[0][0]

        x1 = max(b[0], 0)
        y1 = max(b[1], 0)
        x2 = min(b[2], image_shape[1])
        y2 = min(b[3], image_shape[0])

        im[y1:y2, x1:x2] = m[(y1 - b[1]):(y2 - b[1]), (x1 - b[0]):(x2 - b[0])]
    return im_mask    
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda": 
        get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
        
    # ---------------------- prepare data loader ------------------------------- #

       
    num_classes = 90 +1
    # -------------------------------------------------------------------------- #

    print(args)
    model = maskrcnn_resnet50(True, num_classes, predict = True).to(device)    
    
    # find all checkpoints, and load the latest checkpoint
    prefix, ext = os.path.splitext(args.ckpt_path)
    # ckpts = glob.glob(prefix + "-*" + ext)
    ckpts = glob.glob(prefix + "*" + ext)
    # ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device) # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        torch.cuda.empty_cache()

    model.eval()
    
    # ------------------------------- train ------------------------------------ #
    image_shape = (600, 600)

    capture = cv2.VideoCapture("D:/WorkSpace/JupyterWorkSpace/DataSet/LANEdevkit/Drive-View-Kaohsiung-Taiwan.mp4")
    ori_image_shape    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while(True):    
        ref, frame = capture.read()    
        if not ref:
            break
        # 格式轉變，BGRtoRGB
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = np.uint8(frame)

        image_data = argu.functional.normalize(frame, mean = (0.485, 0.456, 0.406), 
                                                      std = (0.229, 0.224, 0.225))       

        image_data = cv2.resize(image_data, (image_shape[1], image_shape[0]))
        images = np.expand_dims(np.transpose(np.array(image_data, dtype='float32'), (2, 0, 1)), 0)
        images = torch.Tensor(images)

        A = time.time()        
        images = images.to(device)
        output = model(images)
        A = time.time() - A 

        if output["labels"].shape[0] != 0:
            box = output['boxes']
            box[:, [0, 2]] = box[:, [0, 2]] * ori_image_shape[1] / image_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] * ori_image_shape[0] / image_shape[0]
            output['boxes'] = box            

            mask = output['masks']
            mask = paste_masks_in_image(mask, box, 1, ori_image_shape)
            output['masks'] = mask

            mask = mask.detach().cpu().numpy()  
            mask = np.transpose(mask, (1, 2, 0))     
            mask = np.uint8(mask)

            box = box.detach().cpu().numpy()  
            label = output["labels"].detach().cpu().numpy()
            for idx, bb in enumerate(box):
                x1, y1, x2, y2 = bb
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
                print("labels", (x1, y1, x2, y2), label[idx])

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow("w", frame)
        c= cv2.waitKey(1) & 0xff 
        if c==27:
            break     

    
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
    
    