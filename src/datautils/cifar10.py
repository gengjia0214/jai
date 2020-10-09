"""
CIFAR-10 Data Packer

Author: Jia Geng
Email: gjia0214@gmail.com | jxg570@miami.edu
"""

import os
import pickle
import random
import numpy as np
from copy import copy
from datautils.datahandler import DataPackerAbstract


class Cifar10Packer(DataPackerAbstract):
    """
    Packed Cifar 10 dataset
    """

    def __init__(self, src_dir: str, size=(32, 32)):
        """
        Constructor
        :param src_dir: cifar10 data src folder, should contains 5 data batch a test batch
        :param size: image size, should be 32, 32
        """

        super().__init__()

        self.mode = 'memory'  # load the data into memory
        self.src_dir = src_dir
        self.img_ids = []
        self.size = size

    def pack(self, prefix='data_batch'):
        """
        Pack the data batch into full batch and put data on memory
        :param prefix: prefix for data batch files
        :return: void
        """

        # make sure no data already in this object
        assert len(self.img_ids) == 0, 'Data was already packed, call reset() to clear all data before pack().'

        # prepare the data src dict
        img_id = -1  # cifar10 does not have img id, use increment index as img id
        self.labels, self.data_src = {}, {}

        # iterate each data batch
        for fname in sorted(os.listdir(self.src_dir)):

            if fname.startswith(prefix):
                # load the byte file
                with open(os.path.join(self.src_dir, fname), 'rb') as data_batch_file:
                    data_dict = pickle.load(data_batch_file, encoding='bytes')
                    labels = data_dict[b'labels']
                    data = data_dict[b'data']

                    # iterate through all data, put it into the data_src
                    for label, img in zip(labels, data):
                        img_id += 1  # update the img_id
                        self.labels[img_id] = label
                        self.data_src[img_id] = img
                        self.img_ids.append(img_id)

        self.to_rgb()

    def reset(self):
        """
        Reset all attributes
        :return: void
        """

        self.__init__(src_dir=self.src_dir)

    def to_rgb(self):
        """
        Convert the flattened data to 3 channel RGB data
        :return: void
        """

        for k, arr in self.data_src.items():
            if len(arr.shape) != 3:
                arr3d = arr.reshape(3, self.size[0], self.size[1]).transpose(1, 2, 0)
                self.data_src[k] = arr3d

    def get_mean_std(self):
        """
        Get the mean and the std of the data (over pixels for each channel)
        :param size: image size, should be 32x32 for cifar10
        :return: mean, std
        """

        dataset = []
        for k, arr in self.data_src.items():
            dataset.append(arr)
        dataset = np.stack(dataset, axis=0)
        isuint8 = dataset.dtype == 'uint8'
        mean = dataset.reshape(-1, 3).mean(axis=0) / 255 if isuint8 else dataset.reshape(-1, 3).mean(axis=0)
        std = (dataset.reshape(-1, 3) / 255).std(axis=0) if isuint8 else dataset.reshape(-1, 3).std(axis=0)

        return mean, std

    def split(self, train_ratio: float, seed: int, use_copy=True):
        """
        TODO: this can be refactored into the abstract class
        Split the data into train and eval for model training
        :param train_ratio: train ratio
        :param seed: random seed
        :param use_copy: whether to use copy
        :return: training Cifar10Packrt object and eval Cifar10Packrt
        """

        # fix seed
        random.seed(seed)

        # calculate the index
        assert 0 < train_ratio < 1.0, 'Training set ratio'
        upper_idx = int(len(self.img_ids) * train_ratio)

        # split
        random.shuffle(self.img_ids)
        random.shuffle(self.img_ids)

        # get the train/eval data
        train_ids, eval_ids = self.img_ids[:upper_idx], self.img_ids[upper_idx:]
        train_data, eval_data = {i: self.data_src[i] for i in train_ids}, {i: self.data_src[i] for i in eval_ids}
        train_labels, eval_labels = {i: self.labels[i] for i in train_ids}, {i: self.labels[i] for i in eval_ids}

        # copy if needed
        if use_copy:
            train_ids, train_data, train_labels = copy(train_ids), copy(train_data), copy(train_labels)
            eval_ids, eval_data, eval_labels = copy(eval_ids), copy(eval_data), copy(eval_labels)

        # create train/eval packer
        train_packer = Cifar10Packer(src_dir=self.src_dir)
        train_packer.img_ids, train_packer.data_src, train_packer.labels = train_ids, train_data, train_labels
        eval_packer = Cifar10Packer(src_dir=self.src_dir)
        eval_packer.img_ids, eval_packer.data_src, eval_packer.labels = eval_ids, eval_data, eval_labels

        return train_packer, eval_packer
