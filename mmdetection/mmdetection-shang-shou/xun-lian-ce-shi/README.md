# 训练、测试

## 命令说明及示例

{% hint style="info" %}
注意命令都在虚拟环境下运行，注意激活虚拟环境（conda activate your\_env）
{% endhint %}

**`python tools/train.py ./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py`**

* `--work-dir`：指定训练保存模型和日志的路径
* `--resume-from`：从预训练模型`chenkpoint`中恢复训练
* `--no-validate`：训练期间不评估checkpoint
* `--gpus`：指定训练使用GPU的数量（仅适用非分布式训练）
* `--gpu-ids`： 指定使用哪一块GPU（仅适用非分布式训练）
* `--seed`：随机种子
* `--deterministic`：是否为CUDNN后端设置确定性选项
* `--options`： arguments in dict
* `--launcher`： {none,pytorch,slurm,mpi} 分布式训练的任务启动器（job launcher），默认值为none表示不进行分布式训练
* `--local_rank`： LOCAL\_RANK
* `--autoscale-lr`# ： automatically scale lr with the number of gpus

```bash
# 单卡，指定输出目录，可用绝对路径
python tools/train.py \
    configs/yolo/yolov3_copy.py
    --work-dir ./work-dirs/my_yolov3

# 指定GPU
python tools/train.py \
    ./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --gpus 1 --no-validate --work-dir my_faster

# 两个GPU训练,指定输出目录
./tools/dist_train.sh configs/yolov3_copy.py 2 --work-dir ./work-dirs/my_yolov3

# 两个GPU训练，指定输出目录，指定日志输出位置——便于查看错误
./tools/dist_train.sh configs/yolov3_copy.py 2 \
    --work-dir ./work-dirs/my_yolov3 \
    >> yolov3.log 2>&1

# 日志输出，训练命令后加上：
>> yolov3.log 2>&1

# test.py [-h] [--work-dir WORK_DIR] [--out OUT] [--fuse-conv-bn]
               [--gpu-ids GPU_IDS [GPU_IDS ...]] [--gpu-id GPU_ID]
               [--format-only] [--eval EVAL [EVAL ...]] [--show]
               [--show-dir SHOW_DIR] [--show-score-thr SHOW_SCORE_THR]
               [--gpu-collect] [--tmpdir TMPDIR]
               [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]]
               [--options OPTIONS [OPTIONS ...]]
               [--eval-options EVAL_OPTIONS [EVAL_OPTIONS ...]]
               [--launcher {none,pytorch,slurm,mpi}] [--local_rank LOCAL_RANK]
               config checkpoint
# 单GPU测试
python tools/test.py \
    configs/yolo/yolov3_d53_mstrain-608_273e_coco_copy.py \
    checkpoints/latest.pth \
    --out results/test_yolov3.out.pkl \
    --eval mAP \
    --show

# 多GPU测试
./tools/dist_test.sh \
    configs/yolo/yolov3_d53_mstrain-608_273e_coco_copy.py \
    checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth \
    2 \
    --out results/test_yolov3.out.pkl \
    --eval mAP \
    --show
    

```

## **修改`./mmdetection/configs`里面采用的模型配置文件**

{% hint style="info" %}
**一般要修改的内容**
{% endhint %}

* **什么数据集**
* **数据dataloader方式，**
* **模型的num\_classes，多少种类，对应的mmdet中的类**
* **学习率lr与batch size，对比训练官方log日志，与samples\_per\_gpu \* GPU个数成正比**
* **total\_epoch，mmdetection，在模型runner->max\_epoch**
* **learning**_**policy(lr**_**config) 学习策略中的step与max\_epoch有关，学习率是动态调整的。**
