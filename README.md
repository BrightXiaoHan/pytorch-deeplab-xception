# 2019 年县域农业大脑AI挑战赛

### 简介
这是一个基于pytorch的[DeepLab-V3-Plus](https://arxiv.org/pdf/1802.02611)实现。用于参加 [2019 年县域农业大脑AI挑战赛](https://tianchi.aliyun.com/competition/entrance/231717/introduction?spm=5176.12281957.1004.6.38b04c2aPDlxbu)


### 安装
建议在Anaconda，python3.6环境下运行

0. 克隆项目:
    ```Shell
    git clone https://github.com/BrightXiaoHan/pytorch-deeplab-xception
    cd pytorch-deeplab-xception
    ```

1. 安装依赖包:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    pip install matplotlib pillow tensorboardX tqdm
    ```
### 训练
按照以下步骤训练模型:

0. 配置你的数据集路径 [mypath.py](./mypath.py).
目录结构如下
    ```
    ├── image_3_predict.png
    ├── image_4_predict.png
    ├── jingwei_round1_md5_20190619.txt
    ├── jingwei_round1_test_a_20190619
    ├── jingwei_round1_train_20190619
    ```

1. 参数说明: (查看全部参数说明 python train.py --help):
    ```Shell
    usage: train.py [-h] [--backbone {resnet,xception,drn,mobilenet}]
                [--out-stride OUT_STRIDE] [--dataset {pascal,coco,cityscapes}]
                [--use-sbd] [--workers N] [--base-size BASE_SIZE]
                [--crop-size CROP_SIZE] [--sync-bn SYNC_BN]
                [--freeze-bn FREEZE_BN] [--loss-type {ce,focal}] [--epochs N]
                [--start_epoch N] [--batch-size N] [--test-batch-size N]
                [--use-balanced-weights] [--lr LR]
                [--lr-scheduler {poly,step,cos}] [--momentum M]
                [--weight-decay M] [--nesterov] [--no-cuda]
                [--gpu-ids GPU_IDS] [--seed S] [--resume RESUME]
                [--checkname CHECKNAME] [--ft] [--eval-interval EVAL_INTERVAL]
                [--no-val]

    ```

2. 运行训练示例脚本
    ```shell
    bash agriculture_scripts/train_agriculture.sh
    ```
3. 生成提交结果
    ```Shell
    python agriculture_scripts/inference.py
    ```    
