# encoding: utf-8
"""
@author:  qianwen
@contact: qianwen2018@ia.ac.cn
"""

import torchvision.transforms as T
import random
import math
# from .transformer import RandomErasing
from timm.data.random_erasing import RandomErasing

# class RandomErasing(object):
#     """ Randomly selects a rectangle region in an image and erases its pixels.
#         'Random Erasing Data Augmentation' by Zhong et al.
#         See https://arxiv.org/pdf/1708.04896.pdf
#     Args:
#          probability: The probability that the Random Erasing operation will be performed.
#          sl: Minimum proportion of erased area against input image.
#          sh: Maximum proportion of erased area against input image.
#          r1: Minimum aspect ratio of erased area.
#          mean: Erasing value.
#     """

#     def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
#         self.probability = probability
#         self.mean = mean
#         self.sl = sl
#         self.sh = sh
#         self.r1 = r1

#     def __call__(self, img):

#         if random.uniform(0, 1) >= self.probability:
#             return img

#         for attempt in range(100):
#             area = img.size()[1] * img.size()[2]

#             target_area = random.uniform(self.sl, self.sh) * area
#             aspect_ratio = random.uniform(self.r1, 1 / self.r1)

#             h = int(round(math.sqrt(target_area * aspect_ratio)))
#             w = int(round(math.sqrt(target_area / aspect_ratio)))

#             if w < img.size()[2] and h < img.size()[1]:
#                 x1 = random.randint(0, img.size()[1] - h)
#                 y1 = random.randint(0, img.size()[2] - w)
#                 if img.size()[0] == 3:
#                     img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
#                     img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
#                     img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
#                 else:
#                     img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
#                 return img

#         return img

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])
        # transform = T.Compose([
        #     T.Resize(cfg.INPUT.SIZE_TRAIN),
        #     T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        #     # T.Pad(cfg.INPUT.PADDING),
        #     # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        #     T.ToTensor(),
        #     normalize_transform,
        #     RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        # ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform