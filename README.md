---
description: 什么MMDetection
---

# MMDetection上手

{% hint style="info" %}
一个深度学习（_**目标检测**_方向）的_**开源库（开源框架）**_

优点：快速上手目标检测

作者：商汤+港中文
{% endhint %}

## 熟悉结构

mmdetection目录结构

_<mark style="color:orange;">**`注释部分`**</mark>_是重要模块，会有详细介绍

```bash
# mmdetection目录结构

mmdetection/
├── CITATION.cff
├── configs // 设置参数，网络训练以及测试参数，base和网络模型的配置文件
├── demo
├── docker
├── docs
├── LICENSE
├── MANIFEST.in
├── mmdet // 提供了经典网络，常用数据集信息
├── model-index.yml
├── pytest.ini
├── README.md
├── README_zh-CN.md
├── requirements
├── requirements.txt
├── resources
├── setup.cfg
├── setup.py
├── tests
└── tools // 一些工具
```

### 1. mmdetection/config

```
configs/
├── albu_example
├── atss
├── autoassign
├── _base_ //基础模型配置信息，dataset/models/schedules, 其他模型可能会用到这里边的代码，后续介绍
│   ├── datasets // 常用数据集配置
│   │   ├── cityscapes_detection.py
│   │   ├── cityscapes_instance.py
│   │   ├── coco_detection.py
│   │   ├── coco_instance.py
│   │   ├── coco_instance_semantic.py
│   │   ├── coco_panoptic.py
│   │   ├── deepfashion.py
│   │   ├── lvis_v0.5_instance.py
│   │   ├── lvis_v1_instance.py
│   │   ├── openimages_detection.py
│   │   ├── voc0712.py
│   │   └── wider_face.py
│   ├── default_runtime.py //默认训练用到的一些信息，比如日志输出情况，checkpoint输出频率,
│   ├── models // 常用模型配置
│   │   ├── cascade_mask_rcnn_r50_fpn.py
│   │   ├── cascade_rcnn_r50_fpn.py
│   │   ├── faster_rcnn_r50_caffe_c4.py
│   │   ├── faster_rcnn_r50_caffe_dc5.py
│   │   ├── faster_rcnn_r50_fpn.py
│   │   ├── fast_rcnn_r50_fpn.py
│   │   ├── mask_rcnn_r50_caffe_c4.py
│   │   ├── mask_rcnn_r50_fpn.py
│   │   ├── retinanet_r50_fpn.py
│   │   ├── rpn_r50_caffe_c4.py
│   │   ├── rpn_r50_fpn.py
│   │   └── ssd300.py
│   └── schedules // 优化器配置，学习策略，max_epoch等，具体参数可参考此处以及官方训练log
│       ├── schedule_1x.py
│       ├── schedule_20e.py
│       └── schedule_2x.py
├── carafe
├── cascade_rcnn
├── cascade_rpn
├── centernet
├── centripetalnet
├── cityscapes
├── common
├── cornernet
├── dcn
├── dcnv2
├── deepfashion
├── deformable_detr
├── detectors
├── detr
├── double_heads
├── dyhead
├── dynamic_rcnn
├── empirical_attention
├── faster_rcnn
├── fast_rcnn
├── fcos
├── foveabox
├── fpg
├── free_anchor
├── fsaf
├── gcnet
├── gfl
├── ghm
├── gn
├── gn+ws
├── grid_rcnn
├── groie
├── guided_anchoring
├── hrnet
├── htc
├── instaboost
├── lad
├── ld
├── legacy_1.x
├── libra_rcnn
├── lvis
├── maskformer
├── mask_rcnn
├── ms_rcnn
├── nas_fcos
├── nas_fpn
├── openimages
├── paa
├── pafpn
├── panoptic_fpn
├── pascal_voc
├── pisa
├── point_rend
├── pvt
├── queryinst
├── regnet
├── reppoints
├── res2net
├── resnest
├── resnet_strikes_back
├── retinanet
├── rpn
├── sabl
├── scnet
├── scratch
├── seesaw_loss
├── selfsup_pretrain
├── solo
├── sparse_rcnn
├── ssd
├── strong_baselines
├── swin
├── timm_example
├── tood
├── tridentnet
├── vfnet
├── wider_face
├── yolact
├── yolo
├── yolof
└── yolox
```

#### config/\_base\_

#### 1. config/\_base\_/datasets

数据集设置（dataset setting）

*   **dataset\_type**: 数据集类型

    > **mmdetection框架含多种数据集格式**
    >
    > * _**`dataset_type`**_说明使用的数据的类型，是_**`mmdet/datasets/`**_中某个类型文件中类的名字_**`mmdet/datasets/init.py`**_中定义了所有_**`dataset_type`**_的名字可继
    > * _**`mmdet/datasets`**_中的自定义数据集_**`mmdet/datasets/custom.py`**_中，建立自己的数据集，**建议复制一份voc.py或者coco.py，重命名修改**

    * VOC标准数据集（VOCDataset）
    * COCO标准数据集（CocoDataset）
    * cityscapes标准数据集（CityscapesDataset）等
* **data\_root**：数据集的根目录
*   **img\_norm\_cfg**: 图像的标准化设置

    > **默认使用的均值方差为Imagenet的均值和方差。**

    **表示为一个字典，包含**

    * 均值（mean）\[数组]，
    * 方差（std）\[数组]，
    * 是否转为rbg（to\_rgb）\[bool] 。
* **train\_pipline**: 训练管道，包含训练时输入图像数据的信息。
* **test\_pipline**: 测试管道，包含测试时输入图像数据的信息。
* **data**: 整体的数据流通信息，整个字典表示训练，验证，测试时的所有数据信息。包括
  * **workers\_per\_gpu:** 读取数据时每个gpu分配的线程数 。
  * **samples\_per\_gpu**: 每个gpu读取的图像数量，该参数和训练时的gpu数量决定了训练时的batch\_size，_<mark style="color:red;">**`samples_per_gpu * GPUs决定训练学习率`**</mark>_
  * **train**：字典。训练时的数据参数设置，包含了训练时使用的数据集类型，以及训练数据的储存位置，标注文件位置，上述的train\_pipline信息等。
  * **val**：字典。训练中验证时的数据参数设置，包含了验证使用的数据集类型，以及验证数据的储存位置，标注文件位置，验证时使用的管道一般同test\_pipline相同。
  * **test**：字典。测试时的数据参数设置，包含了测试时使用的数据集类型，以及测试数据的储存位置，标注文件位置，上述的test\_pipline信息等。
  * **evaluation**：一个字典，包含验证测试的评价指标等信息。

#### 2. config/\_base\_/models

包含了一些基础模型框架的文件。包含_**`faster-rcnn，yolo`**_等。

一般而言，每一个文件（如：`~/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py`）中包含三个部分：_**模型设置，训练/测试设置，数据集设置，优化器设置，学习策略**_

* **model settings**：模型设置
  * **model：**完成模型的搭建。模型的_**各种组件**_储存在_<mark style="color:orange;">**`mmdet/models`**</mark>_文件夹下。
    * _如果没有看到**`model(字典形式)`**如果没有则可以去首行\_base\_中路径找到。_
* **training and testing settings**
  * **train\_cfg**：训练时的各种阈值设置，
    * 一阶段检测器，
      * assigner(字典格式)
      * allowborder（允许bbox周围扩充像素，0表示不允许）
      * pos\_weight（正样本权重，-1表示不改变原始权重）
      * debug（False表示不是debug模式）
    * 二阶段检测器有所不同，如fast\_rcnn
      * rpn
      * rcnn
  * **test\_cfg**：测试时的各种阈值设置。
* **dataset settings**
* **optimizer**
* **learning policy**

_**models补充：**_

mmdetection框架将检测模型分成了SingleStage和TwoStage两种类型的检测器，各种模型均继承自这两个类完成搭建。&#x20;

**model** 包含type+一阶段或者二阶段的模块。

* **type:** 为模型的类型，预设在mmdet/models/detector文件夹下。除此之外，SinglesStage和TwoStage包含的组件各不相同。
* **SingeStage:** 一阶段检测器表示检测器中没有单独propose检测框的结构（region proposal network），SingleStage包含三大组件：backbone，neck以及bbox_head。_
* **TwoStage:** 二阶段检测器表示模型中单独存在一个组件来做初始的区域筛选。在mmdetection中，二阶段检测器包含五大组件：backbone，neck，rpn\_head_，_bbox\_roi\_extractor，roi\_head。组件均为字典。

#### 3. config/\_base\_/schedules

这个文件夹共三个文件，主要为三种不同的模型超参数的设置（优化器，学习策略，训练epoch），三个文件各不相同，但基本结构一样。

#### 4. 其他文件夹

其他文件夹为各个模型在各种数据集下的具体设置文件，均通过包含上述base文件夹下的三个文件来实现（部分没有包含全部的三个文件，则需要在该文件中重新实现未包含的部分）。

在三个文件全部包含的情况下，只需要进行一点细微设置即可完成全部的训练测试设置。具体的文件通过被tools/train.py或者tools/test.py调用，来完成整个模型的设计以及全部超参数的设置。

### 2. mmdetection/mmdet

{% hint style="info" %}
如果只是想训练模型跑起来，使用标准数据集，这个模块不需要动，直接用即可

<mark style="color:orange;">**如果要修改网络结构，使用自定义数据集再考虑修改mmdet内容**</mark>
{% endhint %}

```vim
mmdet/
├── apis
├── core 
│   ├── anchor
│   ├── bbox
│   │   ├── assigners
│   │   ├── coder
│   │   ├── iou_calculators
│   │   ├── match_costs
│   │   └── samplers
│   ├── data_structures
│   ├── evaluation
│   ├── export
│   ├── hook
│   ├── mask
│   ├── post_processing
│   ├── utils
│   └── visualization
├── datasets
│   ├── api_wrappers
│   ├── pipelines
│   └── samplers
├── models
│   ├── backbones
│   ├── dense_heads
│   ├── detectors
│   ├── losses
│   ├── necks
│   ├── plugins
│   ├── roi_heads
│   │   ├── bbox_heads
│   │   ├── mask_heads
│   │   ├── roi_extractors
│   │   └── shared_heads
│   ├── seg_heads
│   │   └── panoptic_fusion_heads
│   └── utils
└── utils

```

### 3. mmdetection/tools

{% hint style="info" %}
_**analysis\_tools**_可以分析训练日志等

_**dateset\_converters**_用于转换数据格式

_**`dist_test.sh/dist_train.sh`**_用于训练和测试调用了同级目录下的_**`test.py/train..py`**_
{% endhint %}

```vim
tools
├── analysis_tools
│   ├── analyze_logs.py
│   ├── analyze_results.py
│   ├── benchmark.py
│   ├── coco_error_analysis.py
│   ├── confusion_matrix.py
│   ├── eval_metric.py
│   ├── get_flops.py
│   ├── optimize_anchors.py
│   ├── robustness_eval.py
│   └── test_robustness.py
├── dataset_converters
│   ├── cityscapes.py
│   ├── images2coco.py
│   └── pascal_voc.py
├── deployment
│   ├── mmdet2torchserve.py
│   ├── mmdet_handler.py
│   ├── onnx2tensorrt.py
│   ├── pytorch2onnx.py
│   ├── test.py
│   └── test_torchserver.py
├── dist_test.sh
├── dist_train.sh
├── misc
│   ├── browse_dataset.py
│   ├── download_dataset.pyv
│   ├── get_image_metas.py
│   └── print_config.py
├── model_converters
│   ├── detectron2pytorch.py
│   ├── publish_model.py
│   ├── regnet2mmdet.py
│   ├── selfsup2mmdet.py
│   ├── upgrade_model_version.py
│   └── upgrade_ssd_version.py
├── slurm_test.sh
├── slurm_train.sh
├── test.py
└── train.py
```

## 怎么用？

需要这么几步：

* [环境安装](mmdetection/mmdetection-shang-shou/huan-jing-an-zhuang.md)
* [标准数据集（以voc为例）](mmdetection/mmdetection-shang-shou/xun-lian-ce-shi/yolov3+voc2007.md#1.voc2007-shu-ju-ji)先用标准数据集示范，放置好目录后，少量调整可开始训练。
* [训练](mmdetection/mmdetection-shang-shou/xun-lian-ce-shi/yolov3+voc2007.md) 链接到YOLOv3+VOC2007直接上手训练

## 补充内容：

{% hint style="warning" %}
_<mark style="color:red;">**若改动框架源代码后，一定要注意重新编译后再使用，比如修改了mmdet/datasets/voc.py**</mark>_

_<mark style="color:red;">**sudo python \~/mmdetection/setup.py**</mark>_
{% endhint %}

### 1. model结构

> **In mmdetection, model components are basically categorized as 4 types:**&#x20;
>
> **mmdetection**主要四部分构成
>
> * **backbone**: usually a FCN network to extract feature maps, e.g., ResNet. **骨干网，全卷积网络用于提取feature map**
> * **neck**: the part between backbones and heads, e.g., FPN, ASPP. **衔接骨干网和头部。**
> * **head**: the part for specific tasks, e.g., bbox prediction候选框的预测 and mask prediction掩膜的预测. **头部用于特定的部分**
> * **roi extractor**: the part for extracting features from feature maps特征映射图, e.g., RoI Align.**感兴趣（region of interest）提取器，用于从特征映射图中提取特征的部分。**
>
> ****
>
> _`mmdetection`_使用以上组件写了一些通用检测管道——_`SingleStageDetector，TwoStageDetector`._
>
> 两者均位于_**`mmdet.models.detectors`**_中，分别在_**`single_stage.py`**_**和**_**`two_stage.py`**_**中实现。**&#x20;
>
> * _`mmdet/models/detectors/single_stage.py`_实现了一个通用的基础单Stage目标检测模型，具体源码解析见源码处。&#x20;
> * _`mmdet/models/detectors/two_stage.py`_实现了一个通用的基础双Stage目标检测模型，具体源码解析见源码处。&#x20;

_<mark style="color:orange;">**实践中的一些注意事项**</mark>_

* 如果实践中修改了mmcv的相关代码，需要到mmcv文件夹下打开终端，激活mmdetection环境，并运行"pip install ."后才会生效（这样修改的代码才会同步到anaconda的mmdetection环境配置文件中）&#x20;
* 若想使用tensorboard可视化训练过程，在config文件中修改log\_config如下：

```
log_config = dict(
    interval=10,                           # 每10个batch输出一次信息
    hooks=[
        dict(type='TextLoggerHook'),       # 控制台输出信息的风格
        dict(type='TensorboardLoggerHook')  # 需要安装tensorflow and tensorboard才可以使用
    ])
————————————————
版权声明：本文为CSDN博主「周月亮」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/syysyf99/article/details/96574325
```

### _**2. loss部分**_

_**mmdet/models/\[dense\_heads, roi\_heads]下**_

_****_

_****_
