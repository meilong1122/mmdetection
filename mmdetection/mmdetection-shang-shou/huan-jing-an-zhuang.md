---
description: 基于CUDA9.1安装
---

# 环境安装

## MMDetection环境

[官方安装说明 get\_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/zh\_cn/get\_started.md) 用作参考

_**【虚拟环境】——【PyTorch】——【MMDet】**_

### 1. 虚拟环境

```bash
# 创建虚拟环境并激活虚拟环境
# 使用conda创建虚拟环境
# -n 指定名称，mmdet(mmdetection简称)是创建的虚拟环境名称
conda create -n mmdet python=3.7 -y

# 激活虚拟环境，进入后即可使用虚拟环境
conda activate mmdet

# 退出虚拟环境, 要在虚拟环境下使用，无需按此处退出
# conda deactivate
# 如果忘记了虚拟环境名字：
# conda env list
```

### 2. PyTorch

**目前**[**pytorch官网**](https://pytorch.org/get-started/locally/)**提供了cuda10.2和cuda11.3**

**查看cuda版本，nvcc -V 与nvidia-smi显示cuda版本不一样，按照nvcc -V版本来安装**

```bash
# nvcc -V 查看cuda编译工具版本
(base) xbsj@xbsj:~$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Nov__3_21:07:56_CDT_2017
Cuda compilation tools, release 9.1, V9.1.85

# nvidia-smi 显示的是英伟达驱动版本，和
(base) xbsj@xbsj:~$ nvidia-smi
Thu Mar 17 13:13:56 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| 45%   36C    P8    19W / 184W |     99MiB /  7979MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:03:00.0 Off |                  N/A |
| 42%   33C    P8    11W / 184W |      1MiB /  7982MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1599      G   /usr/lib/xorg/Xorg                 39MiB |
|    0   N/A  N/A      3488      G   /usr/bin/gnome-shell               57MiB |
+-----------------------------------------------------------------------------+

# nvcc -V 版本是9.1
# nvidia-smi 版本是11.4
```

更新CUDA参照：[【CUDA】nvcc和nvidia-smi显示的版本不一致？](https://www.jianshu.com/p/eb5335708f2a)

#### **安装**

> [pytorch旧版本链接](https://pytorch.org/get-started/previous-versions/)
>
> * 不建议使用conda安装，推荐pip
> * 由于本地电脑的cuda不是在/usr/local/cuda下，所以官网的教程也无法使用。
> * torch安装按照下方版本安装的
>
> nvcc -V 显示的是9.1，cuda编译版本应该是9.1，所以pytorch版本要对应，不宜过高。
>
> ```bash
> # CUDA 9.2
> pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
> ```

#### **验证**

> ```bash
> (base) xbsj@xbsj:~$ python
> Python 3.8.5 (default, Sep  4 2020, 07:30:14) 
> [GCC 7.3.0] :: Anaconda, Inc. on linux
> Type "help", "copyright", "credits" or "license" for more information.
> >>> import torch
> >>> torch.__version__
> '1.11.0+cu113'
> >>> torch.cuda.is_available()
> True
> ```

### 3. mmdet

**按官网要求安装**

```bash
# (base) xbsj@xbsj:~$ pip install openmin
# bash下输入下面两行
pip install openmin
min install mmdet
```

**下载mmdetection文件，以便使用其中代码**

```bash
# 文件会下载到/home/xbsj
(base) xbsj@xbsj:~$ pwd
/home/xbsj
git clone https://github.com/open-mmlab/mmdetection.git

# 如果git clone非常慢
cd /home/xbsj/下载
# 拷贝到上级目录，..可以替换为绝对路径
cp mmdetection.tar ..
# 解包文件到当前目录下，文件夹名字为mmdetection
tar xvf mmdetection.tar
```

#### 验证环境是否成功

```python
# 文件名：verify_env.py
# python3 
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
inference_detector(model, 'demo/demo.jpg')
```

```bash
# pwd ---> /home/xbsj/mmdetection/ # 当前目录
# 创建 verify_env.py，如下，使用vim。如果不会可以使用vscode远程连接后，手动创建文件
vim verify_env.py
# 写入代码，保存退出
Esc -> :wq -> Enter(回车)
```

**注意 ！！！**

* 如果没有faster\_rcnn\_r50\_fpn\_1x\_coco\_20200130-047c8118.pth需要手动下载。
* 当前目录是/home/xbsj/mmdetection
* 需要手动创建checkpoint

```bash
mkdir checkpoints
cd checkpoints
wget http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
```

***
