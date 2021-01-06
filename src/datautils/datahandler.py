"""
Data handling classes

Author: Jia Geng
Email: gjia0214@gmail.com | jxg570@miami.edu
"""

import os
import torch
import numpy as np
import random
from PIL import Image
from torchvision.transforms.transforms import *
from torch.utils.data.dataloader import *
from torch.utils.data.dataset import *


def default_preprocessing(size: tuple,
                          mean=(0.49139968, 0.48215841, 0.44653091),
                          std=(0.24703223, 0.24348513, 0.26158784),
                          ):
    """
    Get the default preprocessing pipeline
    ToPIL -> Resize -> ToTensor -> Normalization
    :return: list of processing module
    """

    return [Resize(size=size), ToTensor(), Normalize(mean=mean, std=std)]


def default_augmentation(p=0.5):
    """
    Get the default augmentation.
    Random Horizontal Flip & RandomVerticalFlip & Random Jitter
    :return: list of augmentation techniques
    """

    # 0.5 chance of applying the flipping
    # 0.5 chance of horizontal flip & 0.5 chance of vertical flip
    # overall 0.5 x 1.0 + 0.5 x 0.25 = 0.625 chance of Not get flipped
    flip = RandomApply([RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5)], p)

    # p/5 chance of get color jittered (disable the jitter on hue as it would require PIL input)
    # less probability for jitter in case it affect the model learning
    jitter = RandomApply([ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0)], p/5)

    return [flip, jitter]


class DataPackerAbstract:
    """
    An abstract class for packing the dataset.
    The abstract will make sure the packed data container class will be compatible with the ImgDataset
    """

    def __init__(self):

        self.mode = None  # mode

        # below should all be dict format {id: xxx, ...}
        self.data_src = None  # data src should contains id & image data or image fp
        self.labels = None  # labels
        self.additional_info = None  # other information of the data such as bbox locations, etc

    def __len__(self):
        """
        Get the length of the data
        :return: length of the data
        """
        
        if self.labels is None:
            raise Exception('Data has not packed yet. Call pack() to pack the data.')

        return len(self.labels)

    def pack(self, *args):
        """
        Pack data together, can be done by either put all data into memory or construct a memo about the data.
        Should support 2 modes:
        - data in memory
        - data in disk
        - additional information should be loaded on memory
        """

        raise NotImplemented()

    def get_packed_data(self):
        """
        Get the packed data src, should return a dictionary that contains:
        - packing mode (so that Dataset object knows what to expect)
        - data src (data or fp) should be a dictionary {id: data src}
        - annotations should be a dictionary {id: data src} id
        """

        # sanity check
        assert isinstance(self.data_src, dict), 'Data source should be a dictionary with key: id'
        assert isinstance(self.labels, dict), 'Labels should be a dictionary with key: id'
        if self.additional_info is not None: assert isinstance(self.labels, dict), 'Annotation should be a dictionary with key: id'
        assert self.mode in ['disk', 'memory'], 'model must be either disk or memory'

        for data_id in self.data_src:
            if self.additional_info is not None and data_id not in self.additional_info:
                raise Exception('Data id={} from data src can not be found in annotation.'.format(data_id))

        output = {'mode': self.mode, 'data': self.data_src, 'labels': self.labels}
        if self.additional_info is not None:
            output['info'] = self.additional_info

        return output

    def split(self, *args):
        """
        Split the packed data into train, eval, test packer
        """

        raise NotImplemented()

    def to_rgb(self):
        """
        Convert the data to RGB 3-channel format
        """

        pass

    def get_mean_std(self):
        """
        Get the mean and std of the data (per channel)
        """

        pass


class ImgDataset(Dataset):
    """
    PyTorch Dataset + image processing & augmentation
    For image processing pipeline & augmentation, ImageDataset will be compatible with the torchvision.transforms.
    Pass a list of transform modules and ImgDataset will Compose it.

    Use .train() or .eval() to turn on or off the augmentation
    """

    def __init__(self, data_packer: DataPackerAbstract, pre_pipeline: list or None, augmentations: list or None):
        """
        Constructor
        :param data_packer: A packed data object. The object class should inherit the PackedDataAbstract
        :param pre_pipeline: pre-processing pipeline, apply to image for both training and testing time. Note that ToTensor is already included,
        no need to pass it again.
        :param augmentations: Data augmentations. If not None, pass a list of augmentation module. At training time, each image will be
        augmented by the provided augmentation technique.
        """

        # data
        self.packed_data = data_packer.get_packed_data()
        self.state = 'train'

        # idx to img_id to make get_item work
        self.idx2id = {i: img_id for i, img_id in enumerate(self.packed_data['data'])}

        # processing func from torchvision
        self.mapping = {}
        self.pre_processing = None
        self.augmentations = None

        # sanity check and put the callables object into pipeline / augmentations
        if pre_pipeline is not None:
            self.pre_processing = self.__sanity_check(pre_pipeline)
            self.pre_processing = Compose(self.pre_processing)
        if augmentations is not None:
            self.augmentations = self.__sanity_check(augmentations)
            self.augmentations = Compose(self.augmentations)

    def __getitem__(self, idx: int):
        """
        Get item method
        :param idx: the data idx
        :return: data and annotations
        """

        # get the image id
        data_id = self.idx2id[idx]

        # get the mode
        mode = self.packed_data['mode']

        # get he image
        if mode == 'disk':
            fp = self.packed_data['data'][data_id]
            img = Image.open(fp=fp)
        elif mode == 'memory':
            img = self.packed_data['data'][data_id]
        else:
            raise Exception('Mode must be either disk or memory but was {}'.format(mode))

        # check type
        if not isinstance(img, Image.Image):
            img = ToPILImage()(img)

        # apply augmentations
        if self.state == 'train' and self.augmentations is not None:
            img = self.augmentations(img)

        # check type
        if not isinstance(img, Image.Image):
            img = ToPILImage()(img)

        # apply preprocessing pipeline
        if self.pre_processing is not None:
            img = self.pre_processing(img)

        # check type
        if isinstance(img, Image.Image):
            img = ToTensor()(img)

        # get the labels
        label = self.packed_data['labels'][data_id]

        output = {'x': img, 'y': label}

        # get the additional annotations, if any
        if 'info' in self.packed_data:
            output['info'] = self.packed_data['info'][data_id]

        return output

    def __len__(self):
        """
        Get the length of the data
        :return: length of the data
        """

        return len(self.idx2id)

    def eval(self):
        self.state = 'eval'

    def train(self):
        self.state = 'train'

    @staticmethod
    def __sanity_check(funcs: list):
        """
        Sanity check on each input function module.
        Put the callable object into pipeline
        :param funcs:
        :return:
        """

        callables = []
        for func in funcs:
            # sanity check
            assert callable(func), 'Function {} is not an callable object.'.format(func)
            # collect
            callables.append(func)

        return callables


class DataHandler:
    """
    Wrapper on the dataloaders
    """

    def __init__(self, train_dataset: ImgDataset or None, eval_dataset: ImgDataset or None,
                 batch_size: int):
        """
        Constructor
        :param train_dataset: training dataset
        :param eval_dataset: evaluation dataset or testing dataset
        :param batch_size: batch size
        """

        self.dataloaders = {'train': None, 'eval': None}
        if train_dataset is not None:
            train_dataset.train()
            self.dataloaders['train'] = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        if eval_dataset is not None:
            eval_dataset.eval()
            self.dataloaders['eval'] = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

    def __getitem__(self, phase: str):
        """
        Get the dataloader by its phase
        :param phase: phase
        :return: dataloader
        """

        return self.dataloaders[phase]
