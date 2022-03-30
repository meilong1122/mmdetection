---
description: VOC数据集转换成COCO数据集
---

# 数据集

## COCO和VOC数据集

{% hint style="info" %}
<mark style="color:red;">**数据集目录结构非常重要！！！**</mark>

**如果磁盘空间不够，数据在磁盘挂载目录，给\~/mmdectection/data下创建一个软链接**
{% endhint %}

### 1. 目录结构概览

```python
# COCO 数据在mmdetection/data/coco下
# VOC 数据集在mmdetection/data/VOCdevkit下
# 目录结构如下
'''
mmdetection
├── mmdet
├── tools
├── configs
├── data
│ ├── coco
│ │ ├── annotations
│ │ ├── train2017
│ │ ├── val2017
│ │ ├── test2017
│ ├── VOCdevkit
│ │ ├── VOC2007
│ │ │ ├── Annotations
│ │ │ ├── JPEGImages
│ │ │ ├── ImageSets
│ │ │ │ ├── Action
│ │ │ │ ├── Layout
│ │ │ │ ├── Segmentation
│ │ │ │ ├── Main
│ │ │ │ │ ├── test.txt # 测试集
│ │ │ │ │ ├── train.txt # 训练集
│ │ │ │ │ ├── trainval.txt
│ │ │ │ │ ├── val.txt # 验证集
│ │ │ ├── count_data_num.py # 统计text/train.txt等数据集的个数
│ │ │ ├── split_dataset.py # 数据集划分脚本
'''
-------------------------------------------------------
```

{% hint style="info" %}
<mark style="color:orange;">1 如果目录结构中的数据已经下载好，直接跳转到训练部分</mark>

<mark style="color:purple;">2 当数据集需要变动的时候，需要了解后续两节</mark>
{% endhint %}

### 2. COCO 数据集说明

* train2017、val2017、test2017里面反别放置训练、验证、测试图片
*   annotations中包含instances\_train.json、instances\_val.json

    *   .json包含“images”,“annotations”,“type”,"categories"等

        * "images"存放每个图像的名字宽高及图像id
        * "annotations"存放对应相同图像id的图像box的四个坐标位置及该框的类别id
        * "categories"则表示每个类别id到该类真实名字的对应关系




* [目录格式](./#1.-mu-lu-jie-gou-gai-lan)

```
# coco 数据集JSON文件格式
{
    "info": info, # dict
     "licenses": [license], # list ，内部是dict
     "images": [image], # list ，内部是dict
     "annotations": [annotation], # list ，内部是dict
     "categories": # list ，内部是dict
 }
```

### 3. VOC 数据集说明

#### 1. JPEGImages 存放图片数据

#### 2. Annotations 存放xml文件，描述图片信息

```
(base) xbsj@xbsj:~/mmdetection/data/VOCdevkit/VOC2007/Annotations$ cat 000001.xml
<annotation>
	<folder>VOC2007</folder>
	<filename>000001.jpg</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>341012865</flickrid>
	</source>
	<owner>
		<flickrid>Fried Camels</flickrid>
		<name>Jinky the Fruit Bat</name>
	</owner>
	<size>
		<width>353</width>
		<height>500</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>dog</name>
		<pose>Left</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>48</xmin>
			<ymin>240</ymin>
			<xmax>195</xmax>
			<ymax>371</ymax>
		</bndbox>
	</object>
	<object>
		<name>person</name>
		<pose>Left</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>8</xmin>
			<ymin>12</ymin>
			<xmax>352</xmax>
			<ymax>498</ymax>
		</bndbox>
	</object>
</annotation>
```

#### 3.ImageSets

* Action下存放的是人的动作（例如running、jumping等等，这也是VOC challenge的一部分）
* Layout下存放的是具有人体部位的数据（人的head、hand、feet等等，这也是VOC challenge的一部分）
* Segmentation下存放的是可用于分割的数据。
* **Main下存放的是图像物体识别的数据，总共分为20类。**\
  **我们主要关注Main下面的文件.**
  * **20个类，每个类有class\_test/class\_train/class\_trainval/class\_val.txt 共80个文件**
  * **剩下4个文件，test.txt/train.txt/trainval/txt/val.txt**

```
(base) xbsj@xbsj:~/mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main$ ls
aeroplane_test.txt      boat_trainval.txt    cat_test.txt           diningtable_trainval.txt  person_test.txt           sofa_trainval.txt
aeroplane_train.txt     boat_val.txt         cat_train.txt          diningtable_val.txt       person_train.txt          sofa_val.txt
aeroplane_trainval.txt  bottle_test.txt      cat_trainval.txt       dog_test.txt              person_trainval.txt       test.txt
aeroplane_val.txt       bottle_train.txt     cat_val.txt            dog_train.txt             person_val.txt            train_test.txt
bicycle_test.txt        bottle_trainval.txt  chair_test.txt         dog_trainval.txt          pottedplant_test.txt      train_train.txt
bicycle_train.txt       bottle_val.txt       chair_train.txt        dog_val.txt               pottedplant_train.txt     train_trainval.txt
bicycle_trainval.txt    bus_test.txt         chair_trainval.txt     horse_test.txt            pottedplant_trainval.txt  train.txt
bicycle_val.txt         bus_train.txt        chair_val.txt          horse_train.txt           pottedplant_val.txt       train_val.txt
bird_test.txt           bus_trainval.txt     cow_test.txt           horse_trainval.txt        sheep_test.txt            trainval.txt
bird_train.txt          bus_val.txt          cow_train.txt          horse_val.txt             sheep_train.txt           tvmonitor_test.txt
bird_trainval.txt       car_test.txt         cow_trainval.txt       motorbike_test.txt        sheep_trainval.txt        tvmonitor_train.txt
bird_val.txt            car_train.txt        cow_val.txt            motorbike_train.txt       sheep_val.txt             tvmonitor_trainval.txt
boat_test.txt           car_trainval.txt     diningtable_test.txt   motorbike_trainval.txt    sofa_test.txt             tvmonitor_val.txt
boat_train.txt          car_val.txt          diningtable_train.txt  motorbike_val.txt         sofa_train.txt            val.txt
```
