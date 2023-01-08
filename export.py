import argparse
import sys
import time
import warnings
import colorsys
sys.path.append('./')  # to run '$ python *.py' files in subdirectories
import copy

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.nn.functional as F

import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import datetime
import logging

import models
import importlib
import onnxruntime as ort
import numpy as np
import cv2
from glob import glob
import os
from PIL import ImageDraw, ImageFont, Image
from scipy.special import softmax

def select_device(net, device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'{net.upper()} 🚀 torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'
    print(s)
    return torch.device('cuda:0' if cuda else 'cpu')

def resize_image(image, size):
    ih, iw, _  = image.shape
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    new_image       = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_CUBIC)
    # new_image = cv2.rectangle(new_image, ((w-nw)//2,0), (nw - ((w-nw)//2),nh), (128,128,128), 15)

    # cv2.imshow("gray bound", new_image)
    # cv2.waitKey(0)
    return new_image, nw, nh


class Post:
    def __init__(self, opt):
        super(Post, self).__init__()
        self.opt = opt
        if opt.net == "yolact":
            from inst_model.yolact.utils.utils_bbox import BBoxUtility
            self.bbox_util = BBoxUtility()
    def process(self, outputs):
        if opt.net == "yolact":
            # del outputs[3]
            outputs = [torch.from_numpy(o) for o in outputs]           

            results = self.bbox_util.decode_nms(outputs, self.opt.anchors, self.opt.conf_thres, self.opt.iou_thres, image_shape, self.opt.traditional_nms)
            
            return results
            # if not type(results[0]) == torch.Tensor: 
            #     print("Not detected!!!")
            #     return None, _, _

            # box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = [x.cpu().numpy() for x in results]
            # return box_thre, class_thre, class_ids, masks_arg, masks_sigmoid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # error
    parser.add_argument('--config', type=str, default="configs.yolact_base" ,help = 'Path to config .opt file. ')

    parser.add_argument('--weights', type=str, default='best_epoch_weights.pth', help='weights path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', default=True, action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--blend', type=bool, default=True, help='iou threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', default=True, action='store_true', help='simplify onnx model')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='conf threshold for NMS')
    parser.add_argument('--traditional_nms', default=False, action='store_true', help='use traditional nms')
    parser.add_argument('--fp16', action='store_true', help='CoreML FP16 half-precision export')

    opt = parser.parse_args()
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    conf = importlib.import_module(opt.config).get_opts(Train=False)
    for key, value in vars(conf).items():
        setattr(opt, key, value)
    opt.weights = os.path.join(opt.out_path, opt.weights)

    #---------------------------------------------------#
    #   画框设置不同的颜色
    #---------------------------------------------------#
    if opt.num_classes <= 81:
        opt.colors = np.array([[0, 0, 0], [244, 67, 54], [233, 30, 99], [156, 39, 176], [103, 58, 183], 
                                [100, 30, 60], [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212], 
                                [20, 55, 200], [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57], 
                                [70, 25, 100], [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34], 
                                [90, 155, 50], [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34], 
                                [98, 55, 20], [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134], 
                                [90, 125, 120], [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234], 
                                [8, 155, 220], [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64], 
                                [198, 75, 20], [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144], 
                                [78, 155, 120], [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134], 
                                [18, 185, 90], [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84], 
                                [130, 115, 170], [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234], 
                                [18, 25, 190], [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], 
                                [155, 0, 0], [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155], 
                                [155, 0, 255], [255, 155, 0], [155, 255, 0], [0, 155, 255], [0, 255, 155], 
                                [18, 5, 40], [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244], [128, 25, 70]], dtype='uint8')
    else:
        hsv_tuples = [(x / opt.num_classes, 1., 1.) for x in range(opt.num_classes)]
        opt.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        opt.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), opt.colors))


    # print(opt)
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.net, opt.device)

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.input_shape).to(device)  # image size(1,3,320,192) iDetection

    print("Load model.")
    model, _ = models.get_model(opt, train_mode=False)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    print("Load model done.") 

    y = model(img)  # dry run

    if False:
        # ONNX export
        try:
            import onnx

            print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
            f = opt.weights.replace('.pth', '.onnx')  # filename
            model.eval()
            output_names = ['classes', 'boxes'] if y is None else ['output']

            dynamic_axes = None
            if opt.dynamic:
                dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                'output': {0: 'batch', 2: 'y', 3: 'x'}}            

            input_names = ['images']
            torch.onnx.export(model, img, f, verbose=False, opset_version=12, 
                            training        = torch.onnx.TrainingMode.EVAL,
                            do_constant_folding = True,
                            input_names=input_names,
                            output_names=output_names,
                            dynamic_axes=dynamic_axes)

            # Checks
            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model

            graph = onnx.helper.printable_graph(onnx_model.graph)
            # print(graph)  # print a human readable model         
            # onnx_graph_path = opt.weights.replace(".pth", ".txt")
            # with open(onnx_graph_path, "w", encoding="utf-8") as f:
            #     f.write(graph)
            

            if opt.simplify:
                try:
                    import onnxsim

                    print('\nStarting to simplify ONNX...')
                    onnx_model, check = onnxsim.simplify(onnx_model)
                    assert check, 'assert check failed'
                except Exception as e:
                    print(f'Simplifier failure: {e}')

            # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
            f = opt.weights.replace('.pth', '_simp.onnx')  # filename
            onnx.save(onnx_model, f)
            print('ONNX export success, saved as %s' % f)


        except Exception as e:
            print('ONNX export failure: %s' % e)
            exit(-1)

        # Finish
        print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))


    # from det_model.yolov5.utils.utils_bbox import DecodeBox
    # f = os.path.join(opt.out_path, "best_epoch_weights_simp.onnx")
    f = os.path.join(opt.out_path, "best_epoch_weights.onnx")
    ort_session = ort.InferenceSession(f)  

    # Test forward with onnx session (test image) 
    video_path      = os.path.join("D:\WorkSpace\JupyterWorkSpace\DataSet\LANEdevkit", "Drive-View-Kaohsiung-Taiwan.mp4")
    capture = cv2.VideoCapture(video_path)

    fourcc  = cv2.VideoWriter_fourcc(*'XVID')
    size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ref, frame = capture.read()

    fps = 0.0
    drawline = False
    post = Post(opt)

    # for frame in glob("test_images/*.jpg") :
    #     frame = cv2.imread(frame)

    while(True):
        t1 = time.time()
        # 讀取某一幀
        ref, frame = capture.read()
        if not ref:
            break
        t1 = time.time()
        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
        old_img     = copy.deepcopy(frame)
        orininal_h  = np.array(frame).shape[0]
        orininal_w  = np.array(frame).shape[1]
        #---------------------------------------------------#
        image_shape = np.array(np.shape(frame)[0:2])          

        new_image       = cv2.resize(frame, opt.input_shape, interpolation=cv2.INTER_CUBIC)
        # new_image, nw, nh  = resize_image(frame, (opt.input_shape[1], opt.input_shape[0]))
        new_image       = np.expand_dims(np.transpose(np.array(new_image, dtype=np.float32)/255, (2, 0, 1)),0)

        outputs = ort_session.run(
            None, 
            {"images": new_image
             },
        )        
        results = post.process(outputs)

        if results[0] is None:
            continue
        box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = [x.cpu().numpy() for x in results]            
        #----------------------------------------------------------------------#
        #   masks_class [image_shape[0], image_shape[1]]
        #   根据每个像素点所属的实例和是否满足门限需求，判断每个像素点的种类
        #----------------------------------------------------------------------#
        masks_class     = masks_sigmoid * (class_ids[None, None, :] + 1) 
        masks_class     = np.reshape(masks_class, [-1, np.shape(masks_sigmoid)[-1]])
        masks_class     = np.reshape(masks_class[np.arange(np.shape(masks_class)[0]), np.reshape(masks_arg, [-1])], [image_shape[0], image_shape[1]])
        
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        scale       = 0.6
        thickness   = int(max((opt.input_shape[0] + opt.input_shape[1]) // np.mean(opt.input_shape), 1))
        font        = cv2.FONT_HERSHEY_DUPLEX
        color_masks     = opt.colors[masks_class].astype('uint8')
        image_fused     = cv2.addWeighted(color_masks, 0.4, old_img, 0.6, gamma=0)

        
        for i in range(np.shape(class_ids)[0]):
            left, top, right, bottom = np.array(box_thre[i, :], np.int32)

            #---------------------------------------------------------#
            #   获取颜色并绘制预测框
            #---------------------------------------------------------#
            color = opt.colors[class_ids[i] + 1].tolist()
            cv2.rectangle(image_fused, (left, top), (right, bottom), color, thickness)

            #---------------------------------------------------------#
            #   获得这个框的种类并写在图片上
            #---------------------------------------------------------#
            class_name  = opt.class_names[class_ids[i]]
            text_str    = f'{class_name}: {class_thre[i]:.2f}'
            text_w, text_h = cv2.getTextSize(text_str, font, scale, 1)[0]
            cv2.rectangle(image_fused, (left, top), (left + text_w, top + text_h + 5), color, -1)
            cv2.putText(image_fused, text_str, (left, top + 15), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

        # image = Image.fromarray(np.uint8(image_fused))
        frame = image_fused
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %.2f"%(fps))
        frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)
        c= cv2.waitKey(1) & 0xff 
        if c==27:
            capture.release()
            break