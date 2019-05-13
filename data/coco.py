from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ipdb

COCO_ROOT = osp.join(HOME, 'data/coco/')
IMAGES = 'images'
ANNOTATIONS = 'annotations'
COCO_API = 'PythonAPI'
INSTANCES_SET = 'instances_{}.json'
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')


def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        # label_map是一个字典，将1-90的label ID对应到1-80
        # {1: 1, 2: 2, 3: 3,……,11: 11, 13: 12,……,89: 79, 90: 80}
        # 在计算loss的时候(multibox_loss.py)，background的label记为0
        self.label_map = get_label_map(osp.join(COCO_ROOT, 'coco_labels.txt'))

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx] label_idx的下标范围为[0, 79]
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, image_set='trainval35k', transform=None,
                 target_transform=COCOAnnotationTransform(), dataset_name='MS COCO'):
        sys.path.append(osp.join(root, COCO_API))
        # 使用COCO数据集时，transform为：SSDAugmentation
        """
        如果想在pycharm的虚拟环境中，为python3.6解释器安装pycocotools，会提示缺少Python.h等文件
        解决方法是：将usr/include/python3.5中的全部文件复制到pycharm项目文件下venv/include文件夹中
        然后　Cloning https://github.com/waleedka/coco (which is @waleedka's fork of the coco API)
        然后在解压后的文件夹中运行：
        ~/PyTorch-YOLOv3-master/venv/bin/python3.6 PythonAPI/setup.py build_ext install
        
        参考文献：https://github.com/matterport/Mask_RCNN/issues/6
        """
        from pycocotools.coco import COCO
        # self.root: '/home/guoyuang/data/coco/images/trainval35k'
        self.root = osp.join(root, IMAGES, image_set)
        self.label_map = get_label_map(osp.join(COCO_ROOT, 'coco_labels.txt'))

        # ipdb.set_trace()

        # self.coco读取了文件：'/home/guoyuang/data/coco/annotations/instances_trainval35k.json'
        self.coco = COCO(osp.join(root, ANNOTATIONS,
                                  INSTANCES_SET.format(image_set)))

        # imgToAnns是一个字典，它的键为image_id，值为该图片所包含的所有annotations
        # 当image_set='trainval35k'时，len(self.ids) = 35185
        # self.ids存储了35185张图片的ID
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        target = self.coco.imgToAnns[img_id]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        # target储存每张图片中的所有gt，EG: target[0]为{'segmentation': [[474.39……'id': 1099077}
        target = self.coco.loadAnns(ann_ids)
        path = osp.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])

        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        # img.shape (480, 640, 3)
        img = cv2.imread(osp.join(self.root, path))
        height, width, _ = img.shape
        if self.target_transform is not None:
            # target_transform将原始的json格式的真值信息转换为[xmin, ymin, xmax, ymax, label_idx]
            # 其中xmin, ymin, xmax, ymax都缩放在了1*1大小的方格中(x/width y/height)，label_idx的下标范围为[0, 79]
            # 此时的target是一个list，长度为一张图片中所有bbox的个数
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            # 此时的target是一个二维array，shape为n*5
            target = np.array(target)

            # self.transform中的各个数据增强函数都是作者自己写的，并没有使用torchvision中的函数
            # self.transform输入的img为:原图数据、原始尺寸、颜色空间为bgr，EG：(500, 333, 3)
            # 输入的target[:, :4]为:缩放在了1*1大小的方格中的xmin, ymin, xmax, ymax
            # 输入的target[:, 4]为: label_idx
            # self.transform输出的img为:增强后的数据(包括正则化……?)、颜色空间为bgr，尺寸为(300, 300, 3)
            # 输出的boxes为： 增强后的boxes，数据增强后，有些boxes会消失(因为在随机图像裁剪过程中，有些boxes会被裁掉)
            # 输出的labels为：与boxes对应的labels，取值范围为[0, 79]
            img, boxes, labels = self.transform(img, target[:, :4],
                                                target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # target：将boxes和labels又重新拼接在了一个列表中
            # EG：array([[0.83215316, 0.81906153, 0.94524972, 0.84846489, 6.]……])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        # permute(2, 0, 1): 深度学习中，图片数据一般保存为：channel * height * width
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]

        # self.coco.loadImgs(img_id) 根据图像的ID，以列表形式返回json文件中"images"字段的信息
        # 例如: [{'license': 2, 'url': 'http://farm7……_z.jpg', 'file_name': 'COCO_val2014_000000000042.jpg',
        # 'height': 478, 'width': 640, 'date_captured': '2013-11-18 09:22:23', 'id': 42}]
        # self.coco.loadImgs(img_id)[0]['file_name']即为：COCO_val2014_000000000042.jpg

        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        bboxes = []
        category_name = []

        for i in range(len(ann_ids)):
            x1, y1, w, h = self.coco.loadAnns(ann_ids)[i]['bbox']
            x2, y2 = round(x1 + w, 2), round(y1 + h, 2)
            bboxes.append([x1, y1, x2, y2])
            category_id = self.coco.loadAnns(ann_ids)[i]['category_id']
            category_name.append(COCO_CLASSES[self.label_map[category_id] - 1])
        gt = [(category_name[i], bboxes[i]) for i in range(len(ann_ids))]

        return img_id, gt

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
