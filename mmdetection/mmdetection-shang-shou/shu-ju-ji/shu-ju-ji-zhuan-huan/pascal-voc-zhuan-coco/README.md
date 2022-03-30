# Pascal VOC 转 COCO

## VOC数据集划分

{% hint style="info" %}
<mark style="color:orange;">数据集如果已经划分好，跳过此部分</mark>

_**`mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main`****目录下包含：**_

* train.txt
* trainval.txt
* test.txt
* val.txt
{% endhint %}

### 划分train/test/val/trainval

**`split_dataset.py`，放到Annotations同级目录**

```python
# /mmdetection/data/VOCDevkit/VOC2007/split_dataset.py

import os
import random

trainval_percent = 0.8
train_percent = 0.8
xmlfilepath = 'Annotations'
txtsavepath = 'ImageSets\Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('ImageSets/Main/trainval.txt', 'w')
ftest = open('ImageSets/Main/test.txt', 'w')
ftrain = open('ImageSets/Main/train.txt', 'w')
fval = open('ImageSets/Main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
```

_**执行后**_`./data/VOCdevkit/VOC2007/ImageSets/Main`目录下，生成四个txt文件：

* train.txt
* trainval.txt
* test.txt
* val.txt

每个txt文件中存储的都是图片的名字（不含图片名字的后缀`.jpg`）

### VOC转COCO脚本及使用

#### **mmdetection 提供了voc数据集转coco的代码**

{% embed url="https://github.com/open-mmlab/mmdetection/blob/master/tools/dataset_converters" %}
dataset convert
{% endembed %}

{% hint style="info" %}
<mark style="color:orange;">需要已经划分好数据集</mark>

python pascal\_voc.py --out-format coco --out-dir 某个输出的目录

* \*\_test.json
* \*\_train.json
* \*\_trainval.json
* \*\_val.json

_**放到对应的目录coco/annotations**_
{% endhint %}

#### \~/mmdetection/data/pascal\_voc.py

```python
# ~/mmdetection/data/pascal_voc.py
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from mmdet.core import voc_classes

label_ids = {name: i for i, name in enumerate(voc_classes())}


def parse_xml(args):
    xml_path, img_path = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        label = label_ids[name]
        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


def cvt_annotations(devkit_path, years, split, out_file):
    if not isinstance(years, list):
        years = [years]
    annotations = []
    for year in years:
        filelist = osp.join(devkit_path,
                            f'VOC{year}/ImageSets/Main/{split}.txt')
        if not osp.isfile(filelist):
            print(f'filelist does not exist: {filelist}, '
                  f'skip voc{year} {split}')
            return
        img_names = mmcv.list_from_file(filelist)
        xml_paths = [
            osp.join(devkit_path, f'VOC{year}/Annotations/{img_name}.xml')
            for img_name in img_names
        ]
        img_paths = [
            f'VOC{year}/JPEGImages/{img_name}.jpg' for img_name in img_names
        ]
        part_annotations = mmcv.track_progress(parse_xml,
                                               list(zip(xml_paths, img_paths)))
        annotations.extend(part_annotations)
    if out_file.endswith('json'):
        annotations = cvt_to_coco_json(annotations)
    mmcv.dump(annotations, out_file)
    return annotations


def cvt_to_coco_json(annotations):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x1,y1,x2,y2
        # left_top
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))
        # left_bottom
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))
        # right_bottom
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))
        # right_top
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))

        annotation_item['segmentation'].append(seg)

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    for category_id, name in enumerate(voc_classes()):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=0)

        bboxes_ignore = ann['bboxes_ignore'][:, :4]
        labels_ignore = ann['labels_ignore']
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1

    return coco

'''
   输出控制——格式，输出路径
   默认输出*.pkl
'''
def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC annotations to mmdetection format')
    parser.add_argument('devkit_path', help='pascal voc devkit path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--out-format',
        default='pkl',
        choices=('pkl', 'coco'),
        help='output format, "coco" indicates coco annotation format')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    devkit_path = args.devkit_path
    out_dir = args.out_dir if args.out_dir else devkit_path
    mmcv.mkdir_or_exist(out_dir)

    years = []
    if osp.isdir(osp.join(devkit_path, 'VOC2007')):
        years.append('2007')
    if osp.isdir(osp.join(devkit_path, 'VOC2012')):
        years.append('2012')
    if '2007' in years and '2012' in years:
        years.append(['2007', '2012'])
    if not years:
        raise IOError(f'The devkit path {devkit_path} contains neither '
                      '"VOC2007" nor "VOC2012" subfolder')
    out_fmt = f'.{args.out_format}'
    if args.out_format == 'coco':
        out_fmt = '.json'
    for year in years:
        if year == '2007':
            prefix = 'voc07'
        elif year == '2012':
            prefix = 'voc12'
        elif year == ['2007', '2012']:
            prefix = 'voc0712'
        for split in ['train', 'val', 'trainval']:
            dataset_name = prefix + '_' + split
            print(f'processing {dataset_name} ...')
            cvt_annotations(devkit_path, year, split,
                            osp.join(out_dir, dataset_name + out_fmt))
        if not isinstance(year, list):
            dataset_name = prefix + '_test'
            print(f'processing {dataset_name} ...')
            cvt_annotations(devkit_path, year, 'test',
                            osp.join(out_dir, dataset_name + out_fmt))
    print('Done!')


if __name__ == '__main__':
    main()
```

#### 使用命令

```bash
# 注意路径 ~/mmdetection/data/pascal_voc.py
'''
    命令解释：
    1. python解释器运行该文件（要在创建的虚拟环境下）
    2. .pascal_voc.py 为转换脚本
    3. 跟的第一文件夹必须是./VOCdevkit文件夹，该文件的目录结构是
    VOCdevkit/
    ├── VOC2007
    │   ├── Annotations
    │   ├── convert_code
    │   ├── ImageSets
    │   │   ├── Layout
    │   │   ├── Main
    │   │   └── Segmentation
    │   ├── JPEGImages
    │   ├── SegmentationClass
    │   └── SegmentationObject
    └── VOCcode
    4. --out-dir是输出目录
    5. 输出的文件类型，指定coco，输出*.json文件
'''

# 命令如下
(open-mmlab) xbsj@xbsj:~/mmdetection/data$ python ./pascal_voc.py ./VOCdevkit --out-dir ./VOCdevkit --out-format coco

```

## 附一个博客数据集转换链接

{% embed url="https://blog.csdn.net/wulele2/article/details/113468646" %}
