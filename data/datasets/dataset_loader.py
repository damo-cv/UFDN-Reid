# encoding: utf-8
import os
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset

import configparser
import numpy as np
import PIL, PIL.Image
import io
import pickle
import random
import torch
import cv2
import math


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img

# # 通过boto3的client方式
# def read_image(image_id):
#     got_img = False
#     while not got_img and '.jpg' in image_id:
#         try:
#             cv_image = cv2.imread(image_id)
#             if cv_image is None or cv_image.shape[0] * cv_image.shape[1] == 0:
#                 image_id = image_id.split('/')[-1].split('.')[0]
#                 break
#             got_img = True
#         except Exception as e:
#             print("catch exception {} of {}. Will redo. Don't worry. Just chill.".format(e, image_id))
#             exit()
#     if '.jpg' in image_id: return cv_image

# def read_image(img_path):
#     """Keep reading image until succeed.
#     This can avoid IOError incurred by heavy IO process."""
#     got_img = False
#     if not osp.exists(img_path):
#         raise IOError("{} does not exist".format(img_path))
#     while not got_img:
#         try:
#             img = Image.open(img_path).convert('RGB')
#             got_img = True
#         except IOError:
#             print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
#             pass
#     return img

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            # img = cv2.imread(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, size=(256, 256), is_train=True):
        self.dataset = dataset
        self.transform = transform
        self.size = size
        self.is_train = is_train
        self.Earsing = RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_id, pid, camid, tid, date, dset, score_map_path, roi = self.dataset[index]

        # read type 1
        cv_img = read_image(image_id)
        # cv_img = cv2.resize(cv_img, tuple(self.size), cv2.INTER_LINEAR)
        # cv_img = PIL.Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


        # read type 2
        # img = read_image(image_id)
        # cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # # p = random.uniform(0, 1)
        # # if p >= 0.5:
        # #     cv_img = cv2.flip(cv_img, 1)
        # # h, w = cv_img.size[:2]
        # cv_img = cv2.resize(cv_img, tuple(self.size), cv2.INTER_LINEAR)
        # img = PIL.Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        
        # # read type 3
        # cv_img = read_image(image_id)
        # try:
        #     cv_img = cv2.resize(cv_img, tuple(self.size), cv2.INTER_LINEAR)
        #     img = PIL.Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        # except Exception as e:
        #     print('something wrong in {}'.format(image_id))

        # read type 4
        # img = read_image_ks(image_id)

        assert self.transform is not None
        # img = self.transform(img)
        img = self.transform(cv_img)
        # if self.is_train:
        #     img = self.Earsing(img)
        _, resize_h, resize_w = img.size()

        if score_map_path is not None and os.path.exists(score_map_path):
            max_wh, min_wh = max(w, h), min(w, h)
            score_map = pickle.load(open(score_map_path, 'rb'))['score_map']
            score_map = np.transpose(score_map, (1,2,0))
            score_map = cv2.resize(score_map, (max_wh, max_wh))
            begin = int((max_wh - min_wh)/2.)
            end = int((max_wh + min_wh)/2.)
            if min_wh == w:
                score_map = score_map[:, begin:end, :]
            else:
                score_map = score_map[begin:end, :, :]
            size_w, size_h = int(resize_w/16), int(resize_h/16)
            score_map = cv2.resize(score_map, (size_w, size_h))
            score_map = np.transpose(score_map, (2,0,1))
            score_map = torch.from_numpy(score_map)
        else: score_map = None

        # if roi is not None:
        #     if roi == [0,0,1,1]:
        #         pass
        #     else:
        #         roi = [resize_w*roi[0]/w, resize_h*roi[1]/h, resize_w*roi[2]/w, resize_h*roi[3]/h]
        #     # if p >= 0.5:
        #     #     roi = [resize_w-roi[2], roi[1], resize_w-roi[0], roi[3]]
        #     roi = np.array(roi)
        #     roi = torch.from_numpy(roi).float()

        return img, pid, camid, tid, image_id, score_map, roi

        # return img, pid, camid, tid, image_id, None, None

if __name__ == '__main__':
    # ks_image = read_image_ks('04d1fb7f2bedb9546897e623ea1f1df1')
    # print(ks_image)

    ks_image = read_image_ks('/mnt/lustrenew/hezhiqun/experiments/Vehicle_Reid/VehicleReID/Strong/data/VeRi/image_train/0571_c015_00083295_0.jpg')
    import springvision as SP
    transform = SP.Compose([
        SP.Resize((224, 224)),
        SP.RandomHorizontalFlip(p=0.5),
        SP.ToTensor(),
        SP.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    earsing = RandomErasing()
    img = transform(ks_image)
    img = earsing(img)
    print(img.size())
