# encoding=utf-8
import os
import openslide
import numpy as np
import cv2

from PIL import Image
from torch.utils.data import Dataset
from mypath import Path


class AgricutureSegmentation(Dataset):
    """
    阿里天池2019 年县域农业大脑AI挑战赛数据集
    https://tianchi.aliyun.com/competition/entrance/231717/introduction?spm=5176.12281957.1004.5.38b04c2aX5YqAp
    """
    NUM_CLASSES = 4

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir("agriculture"),
                 split="train"
                 ):
        """
        :param base_dir: 县域农业大脑AI挑战赛数据集路径
        :param split: train/val/test
        """
        super().__init__()
        self._base_dir = base_dir
        



