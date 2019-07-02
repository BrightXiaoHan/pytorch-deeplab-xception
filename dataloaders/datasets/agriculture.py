# encoding=utf-8
import os
import openslide
import copy
import cv2
import glob
import shutil
import numpy as np

from PIL import Image
from matplotlib.image import imsave
from tqdm import tqdm
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from dataloaders import custom_transforms as tr
from mypath import Path


Image.MAX_IMAGE_PIXELS = 4000000000  # 设置最大图片像素大小，防止openslide读取图片出错


class AgricutureSegmentation(Dataset):
    """
    阿里天池2019 年县域农业大脑AI挑战赛数据集
    https://tianchi.aliyun.com/competition/entrance/231717/introduction?spm=5176.12281957.1004.5.38b04c2aX5YqAp
    """
    NUM_CLASSES = 4

    def _prepare_data(self):
        """
        预处理。将原始图像裁剪并保存
        """
        images_dir = glob.glob(os.path.join(
            self._base_dir, "*%s*" % self._split))[0]  # 原始数据集存储图像的目录

        work_dir = os.path.join(self._base_dir, "%s_stride%d_imsize%d" % (
            self._split, self.stride, self.img_size))  # 生成的裁剪训练集图片存放目录

        flag_file = os.path.join(work_dir, "finish_flag")  # 标记数据集是否预加载完成

        if os.path.isdir(work_dir) and not os.path.exists(flag_file):
            shutil.rmtree(work_dir)

        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)
            image_list = os.listdir(images_dir)
            image_list.sort()

            # 生成训练集
            if self._split == "train":
                for index in range(0, len(image_list), 2):
                    image = image_list[index]
                    mask = image_list[index + 1]
                    slide_image = openslide.open_slide(
                        os.path.join(images_dir, image))  # 使用openslide打开训练集图片
                    slide_mask = openslide.open_slide(os.path.join(
                        images_dir, mask))  # 使用openslide打开训练集标签图片

                    self._convert(slide_image=slide_image,
                                  image_dir=os.path.join(work_dir, "image"),
                                  tag=image.split('.')[0],  # tag 为去除.png后缀的文件名
                                  slide_mask=slide_mask,
                                  mask_dir=os.path.join(work_dir, 'mask')
                                  )
            # 生成测试集
            elif self._split == "test":
                for index in range(0, len(image_list)):
                    image = image_list[index]
                    slide_image = openslide.open_slide(
                        os.path.join(images_dir, image))  # 使用openslide打开训练集图片

                    self._convert(slide_image=slide_image,
                                  image_dir=os.path.join(work_dir, "image"),
                                  tag=image.split('.')[0]  # tag 为去除.png后缀的文件名
                                  )
            else:
                raise NotImplementedError

        # 生成数据加载完毕的标志文件，再次初始化的时候不需要重新加载
        with open(flag_file, 'w') as f:
            f.write("finish loading.")
        self._work_dir = work_dir

    def _convert(self,
                 slide_image,
                 image_dir,
                 tag,
                 slide_mask=None,
                 mask_dir=None):
        """将原始数据集中的大图裁剪成可以训练的小图并保存

        Args:
            slide_image (ImageSlide): 方法openslide.open_slide的返回值
            img_dir (str): 裁剪后的原始图像的存放路径
            tag (str): 标签字符串，用来标记当前裁剪的小图来自哪一张图片（用于image与mask的对应关系）
            slide_mask (ImageSlide, optional): 方法openslide.open_slide的返回值
            mask_dir (str, optional): 裁剪后的标签小图的存放路径
        """

        width, height = slide_image.dimensions

        # 目录如果不存在则创建目录
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)

        if mask_dir is not None and not os.path.isdir(mask_dir):
            os.makedirs(mask_dir)

        def crop_and_save(w_s, h_s):
            """根据定点坐标裁剪并保存图像

            Args:
                w_s (int): 图片的左上定点坐标
                h_s (int): 图片右上的定点坐标
            """
            img = np.array(slide_image.read_region(
                (w_s, h_s), 0, (self.img_size, self.img_size)))[:, :, :3]
            # 图像占比0.25才保存图像, 测试集图片全部保存
            if np.count_nonzero(img) / (self.img_size * self.img_size) < 0.25 and self._split != 'test':
                return

            image_name = tag + "_" + str(w_s) + "_" + str(h_s)

            if slide_mask is not None:
                mask_label = np.array(slide_mask.read_region(
                    (w_s, h_s), 0, (self.img_size, self.img_size)))

                imsave(os.path.join(
                    mask_dir, image_name+".png"), mask_label)

            # 保存切片图像
            imsave(os.path.join(image_dir, image_name+".jpg"), img)

        for w_s in tqdm(range(0, width, self.stride)):
            for h_s in range(0, height, self.stride):
                crop_and_save(w_s, h_s)

    def __init__(self,
                 args,
                 stride,
                 img_size,
                 base_dir=Path.db_root_dir("agriculture"),
                 split="train"
                 ):
        """
        Args:
            stride (int): 裁剪大图的步长
            img_size (int): 裁剪图片的大小
            base_dir (str, optional): 农业大脑数据集存储目录 Defaults to Path.db_root_dir("agriculture").
            split (str, optional): 训练集或者测试集 Defaults to "train".
        """
        super().__init__()
        self._base_dir = base_dir
        self._split = split
        self.stride = stride
        self.img_size = img_size

        self._prepare_data()

        self.images = [os.path.join(self._work_dir, name) for name in os.listdir(
            os.path.join(self._work_dir, "image"))]
        self.images.sort()

        self.self.categories = None
        if split == "train":
            self.categories = [os.path.join(self._work_dir, name) for name in os.listdir(
                os.path.join(self._work_dir, "mask"))]
            self.categories.sort()
            assert len(self.images) == len(
                self.categories), "Please ensure images and mask is One-to-one correspondence."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self._split == 'val':
            return self.transform_val(sample)
        elif self._split == 'test':
            return self.transform_ts(sample)
        else:
            raise NotImplementedError

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')

        if self.categories is not None:
            _target = Image.open(self.categories[index])

        else:
            _target = copy.deepcopy(_img)

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size,
                               crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        sample = composed_transforms(sample)
        sample.pop("label")
        return sample

    def split(self, ratio=0.7):
        """将训练集数据分为训练集和开发集合

        Args:
            ratio (float): 训练集的占比
        """
        if self._split != 'train':
            raise NotImplementedError

        train_num = int(len(self) * ratio)
        val_num = len(self) - train_num
        train, dev = random_split(self, [train_num, val_num])

        # hack here. 防止训练集和测试集之间产生冲突
        dev_dataset = copy.deepcopy(self)
        dev_dataset._split = "dev"
        dev.dataset = dev_dataset

        return train, dev
