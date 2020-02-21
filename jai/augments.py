"""
Image Augmentation Classes
"""

from torch import Tensor
from PIL import Image
import numpy as np
import random
import torch
import math


class Augmentator:
    """
    Super Class
    """

    def __init__(self, p_start, p_end, *args):
        """
        Constructor
        :param p: probability of applying the augmentation
        """

        self.phase = None
        self.p_start = p_start
        self.p_end = p_end
        self.p = p_start

    def switch_phase(self, phase):
        self.phase = phase

    def step(self, curr_epoch, n_epoch, mode='linear'):
        """
        Take one step to update the probability
        :param curr_epoch: current epoch
        :param n_epoch: total epoch for training
        :param mode: step function
        :return:
        """

        if mode == 'linear':
            self.p = self.p_start + (self.p_end-self.p_start) * curr_epoch / n_epoch
        else:
            raise NotImplemented("Currently only support the linear step.")

    def proc(self, img):
        raise NotImplemented("")

    @staticmethod
    def dummy(x):
        return x


class FuncAugmentator(Augmentator):
    """
    Function based augment that apply any defined function/method on the data
    """

    def __init__(self, p_start, p_end, func):
        """
        Constructor. Pass augmentation function.
        :param p_start: lower bound probability of applying the augmentation
        :param p_end: upper bound probability of applying the augmentation
        :param func: function that will be applied to the image
        """

        super().__init__(p_start, p_end)
        self.func = func

    def proc(self, img):
        """
        Augment the image
        :param img: input image
        :return: augmented image (or without augmentation)
        """

        if self.phase == 'train' and random.random() < self.p:
            return self.func(img)
        else:
            return self.dummy(img)


class AugF:
    """
    Some augmentation function
    """

    @staticmethod
    def grid_mask(img: Tensor, d1: int = 94, d2: int = 244, rotate=90, r=0.5, mode=0, device=torch.device('cpu')):
        """
        Modified version of GridMask Augmentation
        ref: https://arxiv.org/abs/2001.04086 by Chen et al., 2020
        orig implementation: https://github.com/akuxcw/GridMask.git

        Definition of Grid Unit:
         ____ ____
        |  gray   |
        |----|    | d
        |blk_|__r_|

        Paper suggested hyperparameters"
        d1 = 94, d2 = 244
        r = 0.6 for ImageNet (complex image) 0.4 for CIFAR (simple image)

        :param img: input image should be in float Tensor and CxHxW dim order
        :param d1: the lower bound of the grid unit. paper recommendation: 96 (for 224 224)
        :param d2: the upper bound of the gird unit. paper recommendation: 244 (fpr )
        :param rotate: rotate angle of the grid mask
        :param r: ratio shorter gray edge in a grid unit
        :param mode: 0 - keep gray area, 1 - keep black area
        :param device: device
        :return: img after applying the grid mask
        """

        h, w = img.shape[1], img.shape[2]
        mask_l = math.ceil((math.sqrt(h*h + w*w))) # mask should be larger than iamge
        d = np.random.randint(d1, d2)  # draw a random d (grid unit length)

        l_gray = math.ceil(d * r)  # length of the gray area

        mask = np.ones((mask_l, mask_l), np.float32)  # full mask
        st_h, st_w = np.random.randint(d), np.random.randint(d)  # distance bt the image corner and the first grid unit

        # populate the mask
        for i in range(-1, mask_l // d + 1):
            s = d * i + st_h
            t = s + l_gray
            s = max(min(s, mask_l), 0)
            t = max(min(t, mask_l), 0)
            mask[s:t, :] = 0

        for i in range(-1, mask_l // d + 1):
            s = d * i + st_w
            t = s + l_gray
            s = max(min(s, mask_l), 0)
            t = max(min(t, mask_l), 0)
            mask[:, s:t] = 0

        # rotate

        r = np.random.randint(rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(mask_l-h)//2:(mask_l-h)//2+h, (mask_l-w)//2:(mask_l-w)//2+w]

        # keep the black area and drop the gray area
        if mode == 1:
            mask = 1 - mask
        mask = torch.from_numpy(mask).float().to(device)
        mask = mask.expand_as(img)  # align the mask with the image
        img = img * mask
        return img













