import os.path as osp

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

import argparse, os
import importlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attribute Learner')
    parser.add_argument('--config', type=str, default="configs.yolact_base" 
    # parser.add_argument('--config', type=str, default="configs.mask_rcnn_base" 
                        ,help = 'Path to config .opt file. Leave blank if loading from opts.py')
    parser.add_argument('--Image_dir', type=str, default="/home/leyan/DataSet/COCO/val2014")
    parser.add_argument('--Json_path', type=str, default="/home/leyan/DataSet/COCO/annotations_trainval2014/annotations/instances_val2014.json")
    parser.add_argument('--classes_path', type=str, default='model_data/coco_classes.txt')
    parser.add_argument("--map_mode", type=int, default=0 , help="evaluate mode")  

    conf = parser.parse_args() 
    opt = importlib.import_module(conf.config).get_opts(Train=False)
    for key, value in vars(conf).items():     
        setattr(opt, key, value)
    
    d=vars(opt)

       
    mode = opt.map_mode  
    get_classes = importlib.import_module("inst_model.%s.utils.utils"%opt.net).get_classes
    get_coco_label_map = importlib.import_module("inst_model.%s.utils.utils"%opt.net).get_coco_label_map
    Make_json = importlib.import_module("inst_model.%s.utils.utils_map"%opt.net).Make_json
    prep_metrics = importlib.import_module("inst_model.%s.utils.utils_map"%opt.net).prep_metrics             
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、计算指标。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅计算指标。
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = opt.map_mode
    #-------------------------------------------------------#
    #   评估自己的数据集必须要修改
    #   所需要区分的类别对应的txt文件
    #-------------------------------------------------------#
    classes_path = opt.classes_path  
    #-------------------------------------------------------#
    #   获得测试用的图片路径和标签
    #   默认指向根目录下面的datasets/coco文件夹
    #-------------------------------------------------------#
    Image_dir     = opt.Image_dir
    Json_path     = opt.Json_path
    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    #   里面存放了一些json文件，主要是检测结果。
    #-------------------------------------------------------#
    map_out_path    = os.path.join(opt.out_path, 'map_out')
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    test_coco       = COCO(Json_path)
    class_names, _  = get_classes(classes_path)
    COCO_LABEL_MAP  = get_coco_label_map(test_coco, class_names)
    
    ids         = list(test_coco.imgToAnns.keys())[:100]

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        print("Load model done.")
        model = opt.Model_Pred(classes_path=opt.classes_path, num_classes=opt.num_classes, \
                            confidence = 0.05, nms_iou = 0.5)
        print("Get predict result.")
        make_json   = Make_json(map_out_path, COCO_LABEL_MAP)
        for i, id in enumerate(tqdm(ids)):
            image_path  = osp.join(Image_dir, test_coco.loadImgs(id)[0]['file_name'])
            image       = Image.open(image_path)

            box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = model.get_map_out(image)
            if box_thre is None:
                continue
            prep_metrics(box_thre, class_thre, class_ids, masks_sigmoid, id, make_json)
        make_json.dump()
        print(f'\nJson files dumped, saved in: \'eval_results/\', start evaluting.')

    if map_mode == 0 or map_mode == 2:
        bbox_dets = test_coco.loadRes(osp.join(map_out_path, "bbox_detections.json"))
        mask_dets = test_coco.loadRes(osp.join(map_out_path, "mask_detections.json"))

        print('\nEvaluating BBoxes:')
        bbox_eval = COCOeval(test_coco, bbox_dets, 'bbox')
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()

        print('\nEvaluating Masks:')
        bbox_eval = COCOeval(test_coco, mask_dets, 'segm')
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()
