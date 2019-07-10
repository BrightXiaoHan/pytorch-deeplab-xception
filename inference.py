import os
import re
import shutil
import torch
import glob
import argparse
import openslide

import numpy as np
import torchvision.transforms as tr

from collections import namedtuple
from PIL import Image

from tqdm import tqdm
from torch.utils.data import DataLoader

from modeling.deeplab import DeepLab
from dataloaders.utils import decode_segmap
from dataloaders.datasets import agriculture

Image.MAX_IMAGE_PIXELS = 4000000000  # 设置最大图片像素大小，防止openslide读取图片出错

parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Inference")
parser.add_argument('--checkpoint', type=str,
                    help='put the path to checkpoint file')
parser.add_argument("--agriculture-cropsize", type=int, default=512,
    help="Image size cropped from large slide.")
parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],)
args=parser.parse_args()


def assemble():

    # 获取原数据集目录，工作目录，步长，图像大小等信息
    test_set=agriculture.AgricutureSegmentation(
        None, parser.agriculture_cropsize, parser.agriculture_cropsize, split='test')
    base_dir=test_set._base_dir
    work_dir=test_set._work_dir
    stride, imsize=re.match(
        "test_stride([0-9]*)_imsize([0-9]*)", os.path.basename(work_dir)).groups()
    stride, imsize=int(stride), int(imsize)

    # 暂时支持步长与图像大小相等的拼接
    assert stride == imsize

    # 获取原始卫星照片，以及文件名前缀
    all_slide=glob.glob(os.path.join(base_dir, "jingwei*test*/*"))
    all_prefix=[os.path.basename(item).split('.')[0] for item in all_slide]
    all_split_images=glob.glob(os.path.join(work_dir, "mask", "*"))

    for prefix, slide in zip(all_prefix, all_slide):
        slide_image=openslide.open_slide(slide)
        width, height=slide_image.dimensions
        slide_mask_buffer=np.zeros(
            (height + imsize, width + imsize), dtype="uint8")

        for part in tqdm(all_split_images):
            base_name=os.path.basename(part)
            if prefix not in base_name:
                continue
            x1, y1=re.match(
                prefix + "_([0-9]*)_([0-9]*).*", base_name).groups()
            x1, y1=int(x1), int(y1)
            x2=x1 + imsize
            y2=y1 + imsize
            part_image=np.asarray(Image.open(
                part).convert('L'), dtype="uint8")

            slide_mask_buffer[y1:y2, x1:x2]=part_image

        slide_mask_buffer=slide_mask_buffer[:height, :width]
        slide_mask_image=Image.fromarray(slide_mask_buffer, 'L')

        save_name=prefix + "_predict.png"

        slide_mask_image.save(os.path.join(work_dir, save_name))

def inference():

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    checkpoint=torch.load(args.checkpoint)

    model=DeepLab(num_classes=4,
                    backbone=args.backbone,
                    output_stride=16,
                    sync_bn=True,
                    freeze_bn=False)

    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()

    kwargs={'num_workers': 0, 'pin_memory': True}
    test_set=agriculture.AgricutureSegmentation(None, 512, 512, split='test')
    test_loader=DataLoader(test_set, batch_size=4, shuffle=False, **kwargs)

    torch.set_grad_enabled(False)

    result=[]

    for inputs in tqdm(test_loader):
        inputs=inputs['image']
        inputs=inputs.cuda()
        output=model(inputs).cpu().numpy()
        pred=np.argmax(output, axis=1).astype("uint8")
        result.append(pred)

    result=np.concatenate(result)
    saved_dir=os.path.join(test_set._work_dir, "mask")
    if os.path.isdir(saved_dir):
        shutil.rmtree(saved_dir)

    os.mkdir(saved_dir)

    for image, name in zip(result, test_set.images):
        image=Image.fromarray(image, 'L')
        name=name.split('/')
        name[-2]="mask"
        name='/'.join(name)
        name=name.replace("jpg", "png")
        image.save(name)


if __name__ == "__main__":
    inference()
    assemble()
