import argparse
import os
import glob
import numpy as np
import json
import xmltodict
import cv2
import PIL.Image
import time
import image_processing, file_processing

'''
Reference:
https://github.com/PanJinquan/python-learning-notes/blob/f45f0879cc70eb59de67a270a6ec8dbb2cf8e742/modules/dataset_tool/coco_tools/convert_voc2coco.py
'''

class SegmentationObject(object):
    """PascalVOC SegmentationObject"""

    @staticmethod
    def change_format(contour):
        contour2 = []
        length = len(contour)
        for i in range(0, length, 2):
            contour2.append([contour[i], contour[i + 1]])
        return np.asarray(contour2, np.int32)

    @staticmethod
    def get_segmentation_area(seg_path, bbox):
        """
        :param seg_path:
        :param bbox: bbox = [xmin, ymin, xmax, ymax]
        :return:seg is [[...],[...]],area is int
        """
        area = 0
        seg = SegmentationObject.getsegmentation(seg_path, bbox)
        if seg:
            seg = [list(map(float, seg))]
            contour = SegmentationObject.change_format(seg[0])
            # 计算轮廓面积
            area = abs(cv2.contourArea(contour, True))
        return seg, area

    @staticmethod
    def getsegmentation(seg_path, bbox):
        """
        :param seg_path:
        :param bbox: bbox = [xmin, ymin, xmax, ymax]
        :return:
        """
        if not os.path.exists(seg_path):
            return []
        try:
            mask_1 = cv2.imread(seg_path, 0)
            mask = np.zeros_like(mask_1, np.uint8)
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask_1[bbox[1]:bbox[3],
                                                     bbox[0]:bbox[2]]

            # 计算矩形中点像素值
            mean_x = (bbox[0] + bbox[2]) // 2
            mean_y = (bbox[1] + bbox[3]) // 2

            end = min((mask.shape[1], int(bbox[2]) + 1))
            start = max((0, int(bbox[0]) - 1))

            flag = True
            for i in range(mean_x, end):
                x_ = i
                y_ = mean_y
                pixels = mask_1[y_, x_]
                if pixels != 0 and pixels != 220:  # 0 对应背景 220对应边界线
                    mask = (mask == pixels).astype(np.uint8)
                    flag = False
                    break
            if flag:
                for i in range(mean_x, start, -1):
                    x_ = i
                    y_ = mean_y
                    pixels = mask_1[y_, x_]
                    if pixels != 0 and pixels != 220:
                        mask = (mask == pixels).astype(np.uint8)
                        break
            polygons = SegmentationObject.mask2polygons(mask)
            return polygons
        except:
            return []

    @staticmethod
    def mask2polygons(mask):
        '''从mask提取边界点'''
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
        bbox = []
        for cont in contours[0]:
            [bbox.append(i) for i in list(cont.flatten())]
            # map(bbox.append,list(cont.flatten()))
        return bbox  # list(contours[1][0].flatten())

    @staticmethod
    def getbbox(height, width, points):
        '''边界点生成mask，从mask提取定位框'''
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points
        mask = SegmentationObject.polygons_to_mask([height, width], polygons)
        return SegmentationObject.mask2box(mask)

    @staticmethod
    def mask2box(mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    @staticmethod
    def polygons_to_mask(img_shape, polygons):
        '''边界点生成mask'''
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask


class CustomVoc():
    # skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
    #             [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    # skeleton = np.asarray(skeleton) - 1

    # skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
    #             (8, 10), (11, 13), (12, 14), (13, 15), (14, 16), (11, 17), (12, 17)]

    skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
                (8, 10), (11, 13), (12, 14), (13, 15), (14, 16), (11, 18), (12, 18), (18, 17)]

    def __init__(self, anno_dir, image_dir=None, seg_dir=None):
        """
        Custom VOC dataset
        :param anno_dir:
        :param image_dir:
        :param seg_dir:
        """
        self.anno_dir = anno_dir
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.xml_list = self.get_xml_files(self.anno_dir)
        self.seg = SegmentationObject()

    @staticmethod
    def get_xml_files(xml_dir):
        """
        :param xml_dir:
        :return:
        """
        xml_path = os.path.join(xml_dir, "*.xml")
        xml_list = glob.glob(xml_path)
        xml_list=sorted(xml_list)
        return xml_list

    @staticmethod
    def read_xml2json(xml_file):
        """
        import xmltodict
        :param xml_file:
        :return:
        """
        with open(xml_file) as fd:  # 将XML文件装载到dict里面
            content = xmltodict.parse(fd.read())
        return content

    def check_image(self, filename, shape: tuple):
        """
        check image size
        :param filename:
        :param shape:
        :return:
        """
        if self.image_dir:
            image_path = os.path.join(self.image_dir, filename)
            assert os.path.exists(image_path), "not path:{}".format(image_path)
            image = cv2.imread(image_path)
            _shape = image.shape
            assert _shape == shape, "Error:{}".format(image_path)

    def get_segmentation_area(self, filename, bbox):
        """
        :param filename:
        :param bbox:[xmin, ymin, xmax, ymax]
        :return:
        """
        seg = []
        area = 0
        if self.seg_dir:
            # if exist VOC SegmentationObject
            seg_path = os.path.join(self.seg_dir, filename.split('.')[0] + '.png')
            seg, area = self.seg.get_segmentation_area(seg_path, bbox)
        if not seg:
            # cal seg and area by bbox
            xmin, ymin, xmax, ymax = bbox
            seg = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]
            area = (xmax - xmin) * (ymax - ymin)
        return seg, area

    @staticmethod
    def get_annotation(xml_file):
        """
        keypoints = object["keypoints"]
        joint = np.asarray(keypoints).reshape(17, 3)
        joint = joint[:, 0:2]
        :param xml_file:
        :return:
        """
        content = CustomVoc.read_xml2json(xml_file)
        annotation = content["annotation"]
        # get image shape
        width = int(annotation["size"]["width"])
        height = int(annotation["size"]["height"])
        depth = int(annotation["size"]["depth"])

        filename = annotation["filename"]
        # self.check_image(filename, shape=(height, width, depth))

        objects_list = []
        objects = annotation["object"]
        if not isinstance(objects, list):
            objects = [objects]
        for object in objects:
            class_name = object["name"]
            xmin = float(object["bndbox"]["xmin"])
            xmax = float(object["bndbox"]["xmax"])
            ymin = float(object["bndbox"]["ymin"])
            ymax = float(object["bndbox"]["ymax"])
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            rect = [xmin, ymin, xmax - xmin, ymax - ymin]
            bbox = [xmin, ymin, xmax, ymax]
            # get person keypoints ,if exist
            if 'keypoints' in object:
                keypoints = object["keypoints"]
                keypoints = [float(i) for i in keypoints.split(",")]
            else:
                keypoints = [0] * 17 * 3
            kp_bbox = {}
            kp_bbox["keypoints"] = keypoints
            kp_bbox["bbox"] = bbox
            kp_bbox["class_name"] = class_name
            objects_list.append(kp_bbox)
            # seg, area = self.get_segmentation_area(filename, bbox=bbox)
        annotation_dict = {}
        annotation_dict["image"] = filename
        annotation_dict["object"] = objects_list
        return annotation_dict

    def decode_voc(self, vis=True):
        """
        :return:
        """
        for xml_file in self.xml_list:
            anns = CustomVoc.get_annotation(xml_file)
            filename = anns["image"]
            object = anns["object"]
            keypoints = []
            bboxes = []
            class_name = []
            for item in object:
                joint = item["keypoints"]
                joint = np.asarray(joint).reshape(17, 3)
                joint = joint[:, 0:2]
                keypoints.append(joint.tolist())
                bboxes.append(item["bbox"])
                class_name.append(item["class_name"])
            image_path = os.path.join(self.image_dir, filename)
            image = image_processing.read_image(image_path)
            self.show(filename, image, keypoints, bboxes, class_name, vis=True)

    def write_to_json(self, json_dir):
        file_processing.create_dir(json_dir)
        for xml_file in self.xml_list:
            anns = self.get_annotation(xml_file)
            name = os.path.basename(xml_file)[:-len(".jpg")]
            json_path = os.path.join(json_dir, name + ".json")
            file_processing.write_json_path(json_path, anns)

    def show(self, filename, image, keypoints, bboxes, class_name, vis=True):
        is_save = True
        for i, joints in enumerate(keypoints):
            if np.sum(np.asarray(joints[5])) == 0 or np.sum(np.asarray(joints[6])) == 0 or \
                    np.sum(np.asarray(joints[11])) == 0 or np.sum(np.asarray(joints[12])) == 0:
                is_save = False
            else:
                is_save = True
            chest_joint = (np.asarray(joints[5]) + np.asarray(joints[6])) / 2
            hip_joint = (np.asarray(joints[11]) + np.asarray(joints[12])) / 2
            keypoints[i].append(chest_joint.tolist())
            keypoints[i].append(hip_joint.tolist())

        if vis:
            image_processing.draw_image_bboxes_text(image, bboxes, class_name)
            # image_processing.show_image_boxes(None, image, joints_bbox, color=(255, 0, 0))
            image = image_processing.draw_key_point_in_image(image, keypoints, pointline=self.skeleton)
            # image_processing.cv_show_image("Det", image, waitKey=0)
            # self.save_images(image, filename, is_save)
            image_processing.cv_show_image("Det", image)

    def save_images(self, image, filename, is_save):
        if is_save:
            out_dir = "/media/dm/dm/project/dataset/COCO/HumanPose/LeXue_teacher/Posture/1"
            out_dir = file_processing.create_dir(out_dir)
            out_image_path = os.path.join(out_dir, filename)
            image_processing.save_image(out_image_path, image)
        else:
            out_dir = "/media/dm/dm/project/dataset/COCO/HumanPose/LeXue_teacher/Posture/unknown/1"
            out_dir = file_processing.create_dir(out_dir)
            out_image_path = os.path.join(out_dir, filename)
            image_processing.save_image(out_image_path, image)


def save_json(data_coco, json_file):
    """
    save COCO data in json file
    :param json_file:
    :return:
    """
    json.dump(data_coco, open(json_file, 'w'), indent=4)  # indent=4 更加美观显示
    print("save file:{}".format(json_file))


def read_json(json_path):
    """
    读取数据
    :param json_path:
    :return:
    """
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data


class COCOTools(object):
    """COCO Tools"""

    @staticmethod
    def get_categories_id(categories):
        """
        get categories id dict
        :param categories:
        :return: dict:{name:id}
        """
        supercategorys = []
        categories_id = {}
        for item in categories:
            supercategory = item["supercategory"]
            name = item["name"]
            id = item["id"]
            categories_id[name] = id
        return categories_id

    @staticmethod
    def get_annotations_id(annotations):
        """
        get annotations id list
        :param annotations:
        :return: annotations id list
        """
        annotations_id = []
        for item in annotations:
            id = item["id"]
            annotations_id.append(id)
        return annotations_id

    @staticmethod
    def get_images_id(images):
        """
        get image id list
        :param images:
        :return: images id list
        """
        images_id = []
        for item in images:
            id = item["id"]
            images_id.append(id)
        return images_id

    @staticmethod
    def check_uniqueness(id_list: list, title="id"):
        """
        检测唯一性
        :return:
        """
        for i in id_list:
            n = id_list.count(i)
            assert n == 1, Exception("have same {}:{}".format(title, i))

    @staticmethod
    def check_coco(coco):
        """
        检测COCO合并后数据集的合法性
            检测1: 检测categories id唯一性
            检测2: 检测image id唯一性
            检测3: 检测annotations id唯一性
        :return:
        """
        categories_id = COCOTools.get_categories_id(coco["categories"])
        print("categories_id:{}".format(categories_id))
        categories_id = list(categories_id.values())
        COCOTools.check_uniqueness(categories_id, title="categories_id")

        image_id = COCOTools.get_images_id(coco["images"])
        COCOTools.check_uniqueness(image_id, title="image_id")

        annotations_id = COCOTools.get_annotations_id(coco["annotations"])
        COCOTools.check_uniqueness(annotations_id, title="annotations_id")
        print("have image:{}".format(len(image_id)))


class PascalVoc2Coco(CustomVoc):
    """Convert Pascal VOC Dataset to COCO dataset format"""

    def __init__(self, anno_dir, image_dir=None, seg_dir=None, init_id=None):
        """
        :param anno_dir:  for voc `Annotations`
        :param image_dir: for voc `JPEGImages`,if image_dir=None ,will ignore checking image shape
        :param seg_dir:   for voc `SegmentationObject`,if seg_dir=None,will ignore Segmentation Object
        :param image_id: 初始的image_id,if None,will reset to currrent time
        """
        self.anno_dir = anno_dir
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.xml_list = self.get_xml_files(self.anno_dir)
        self.seg = SegmentationObject()

        self.coco = dict()
        self.coco['images'] = []
        self.coco['type'] = 'instances'
        self.coco['annotations'] = []
        self.coco['categories'] = []

        self.category_set = dict()
        self.image_set = set()

        self.category_item_id = 0
        if not init_id:
            init_id = int(time.time()) * 2
        self.image_id = init_id
        # self.image_id = 20200207
        self.annotation_id = 0

    def addCatItem(self, name):
        """
        :param name:
        :return:
        """
        self.category_item_id += 1
        category_item = dict()
        category_item['supercategory'] = name
        category_item['id'] = self.category_item_id
        category_item['name'] = name
        if name == "person":
            category_item['keypoints'] = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
                                          'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                                          'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                                          'right_ankle']
            category_item['skeleton'] = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
                                         [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4],
                                         [3, 5], [4, 6], [5, 7]]

        self.coco['categories'].append(category_item)
        self.category_set[name] = self.category_item_id
        return self.category_item_id

    def addImgItem(self, file_name, image_size):
        """
        :param file_name:
        :param image_size: [height, width]
        :return:
        """
        if file_name is None:
            raise Exception('Could not find filename tag in xml file.')
        self.image_id += 1
        image_item = dict()
        image_item['id'] = self.image_id
        image_item['file_name'] = file_name
        image_item['height'] = image_size[0]
        image_item['width'] = image_size[1]
        self.coco['images'].append(image_item)
        self.image_set.add(file_name)
        return self.image_id

    def addAnnoItem(self, image_id, category_id, rect, seg, area, keypoints):
        """
        :param image_id:
        :param category_id:
        :param rect:[x,y,w,h]
        :param seg:
        :param area:
        :param keypoints:
        :return:
        """
        self.annotation_id += 1
        annotation_item = dict()
        annotation_item['segmentation'] = seg
        annotation_item['area'] = area
        annotation_item['iscrowd'] = 0
        annotation_item['ignore'] = 0
        annotation_item['image_id'] = image_id  #
        annotation_item['bbox'] = rect  # [x,y,w,h]
        annotation_item['category_id'] = category_id
        annotation_item['id'] = self.annotation_id
        annotation_item['num_keypoints'] = int(len(keypoints) / 3)
        # annotation_item['keypoints'] = keypoints
        self.coco['annotations'].append(annotation_item)

    def generate_dataset(self):
        """
        :return:
        """
        for xml_file in self.xml_list:
            # convert XML to Json
            content = self.read_xml2json(xml_file)
            annotation = content["annotation"]
            # get image shape
            width = int(annotation["size"]["width"])
            height = int(annotation["size"]["height"])
            depth = int(annotation["size"]["depth"])

            filename = annotation["filename"]
            self.check_image(filename, shape=(height, width, depth))
            if filename in self.category_set:
                raise Exception('file_name duplicated')

            if filename not in self.image_set:
                image_size = [height, width]
                current_image_id = self.addImgItem(filename, image_size=image_size)
                print('add filename {}'.format(filename))
            else:
                raise Exception('duplicated image_dict: {}'.format(filename))

            objects = annotation["object"]
            if not isinstance(objects, list):
                objects = [objects]
            for object in objects:
                class_name = object["name"]
                if class_name not in self.category_set:
                    current_category_id = self.addCatItem(class_name)
                else:
                    current_category_id = self.category_set[class_name]
                xmin = float(object["bndbox"]["xmin"])
                xmax = float(object["bndbox"]["xmax"])
                ymin = float(object["bndbox"]["ymin"])
                ymax = float(object["bndbox"]["ymax"])
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                rect = [xmin, ymin, xmax - xmin, ymax - ymin]
                bbox = [xmin, ymin, xmax, ymax]

                # get person keypoints ,if exist
                if 'keypoints' in object:
                    keypoints = object["keypoints"]
                    keypoints = [float(i) for i in keypoints.split(",")]
                else:
                    keypoints = [0] * 17 * 3
                # get segmentation info
                seg, area = self.get_segmentation_area(filename, bbox=bbox)
                if area > 0:
                    self.addAnnoItem(current_image_id, current_category_id, rect, seg, area, keypoints)
        COCOTools.check_coco(self.coco)

    def get_coco(self):
        return self.coco

    def save_coco(self, json_file):
        save_json(self.get_coco(), json_file)


def main():
    parser = argparse.ArgumentParser(description="COCO Dataset")
    parser.add_argument("-i", "--image_dir", help="path/to/image", type=str)
    parser.add_argument("-a", "--anno_dir", help="path/to/anno_dir", type=str)
    parser.add_argument("-seg_dir", "--seg_dir", help="path/to/VOC/SegmentationObject", default=None, type=str)
    parser.add_argument("-s", "--save_path", help="out/to/save_json-file", type=str)
    parser.add_argument("-id", "--init_id", help="init id", type=int, default=None)
    args = parser.parse_args()

    image_dir = args.image_dir
    anno_dir = args.anno_dir
    seg_dir = args.seg_dir
    save_path = args.save_path
    init_id = args.init_id

    anno_dir = 'D:\WorkSpace\JupyterWorkSpace\DataSet\VOCdevkit\VOC2007\Annotations'  # 这是xml文件所在的地址
    seg_dir = "D:\WorkSpace\JupyterWorkSpace\DataSet\VOCdevkit\VOC2007\SegmentationObject"
    image_dir = "D:\WorkSpace\JupyterWorkSpace\DataSet\VOCdevkit\VOC2007\JPEGImages"
    json_file = 'D:\WorkSpace\JupyterWorkSpace\DataSet\VOCdevkit\VOC2007\Annotations\VOC2007.json'  # 这是你要生成的json文件

    VOC2coco = PascalVoc2Coco(anno_dir, image_dir=image_dir, seg_dir=seg_dir, init_id=init_id)
    VOC2coco.generate_dataset()
    VOC2coco.save_coco(json_file)

    anno_dir = 'D:\WorkSpace\JupyterWorkSpace\DataSet\VOCdevkit\VOC2012\Annotations'  # 这是xml文件所在的地址
    seg_dir = "D:\WorkSpace\JupyterWorkSpace\DataSet\VOCdevkit\VOC2012\SegmentationObject"
    image_dir = "D:\WorkSpace\JupyterWorkSpace\DataSet\VOCdevkit\VOC2012\JPEGImages"
    json_file = 'D:\WorkSpace\JupyterWorkSpace\DataSet\VOCdevkit\VOC2012\Annotations\VOC2012.json'  # 这是你要生成的json文件

    VOC2coco = PascalVoc2Coco(anno_dir, image_dir=image_dir, seg_dir=seg_dir, init_id=init_id)
    VOC2coco.generate_dataset()
    VOC2coco.save_coco(json_file)

if __name__ == '__main__':
     main()
