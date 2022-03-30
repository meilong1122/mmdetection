# 自定义数据集

{% hint style="info" %}
<mark style="color:orange;">自定义数据集修改的地方比较多，</mark>

<mark style="color:orange;">目录格式，文件序号顺序命名，存储位置。尽可能符合标准数据集的样式，这样修改处要少一些。</mark>

所有文件如果要覆盖，建议先备份，再修改，因为mmdet下载到了本地，修改后调用会出问题。

* 命令：cp 源文件 源文件.backup

文件中追加自定义的的方法、类没有问题。

* mmdet/datasets/voc.py或者coco.py
{% endhint %}

## 修改数据集的类别classes

* 在**`./mmdetection/mmdet/core/evaluation/class_names.py`**自定义方法，返回类别
* 在**`mmdetection/mmdet/datasets/[voc.py, coco.py]`**中调用_**`class_names.py`**_中的自定义方法修改**`CLASSES。`**

{% hint style="info" %}
**有博客提示：当仅含有一个类别时，要加一个逗号，根据实际情况判断。**

_<mark style="color:red;">`IndentationError: unexpected indent`</mark>_

<mark style="color:orange;">原因分析：</mark>

* CLASSES应该是一个可迭代的对象，类型是元祖。如果用了_**（）**_，且_**（）**_内是一个字符串，当迭代（‘one\_class’）时，其实是迭代了字符串（该类型是字符串而不是元祖，括号相当于没有）。（'one\_class',）加一个逗号后，迭代的是元祖，长度为1的元祖。
* 有博主说：旧代码版本`num_classes + 1`，把背景算作一类，在`mmdetection V2.0.0`版本，背景不再作为一类，因此不再加1。
{% endhint %}

### 测试修改：

如果使用voc2007，_**`tool/my_voc_eval.py`**_文件，需要注释掉判断voc数据集类型的代码：

![](<../../../.gitbook/assets/image (1).png>)

需要修改_**`dataset_name = dataset.CLASSES`**_

## 修改其他配置

[参照3+VOC2007处的修改](yolov3+voc2007.md#2.-xiu-gai)
