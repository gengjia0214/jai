import os
import csv
import torch
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from torchvision import utils
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.dataset import random_split


def extract_img_from_parquet(src_dir, dst_dir):
    """
    Extract the images from the parquet files.
    Apply to train/test data separately.
    :param src_dir: src folder contains parquet images.
    :param dst_dir: dst folder for placing the extracted images. Make sure it is clean.
    :return: void
    """

    curr_folder = os.path.join(dst_dir, "0")
    os.mkdir(curr_folder)
    for file in sorted(os.listdir(src_dir)):
        if file.endswith('.parquet'):
            df = pd.read_parquet(os.path.join(src_dir, file))
            print("Total of {} images in {}.".format(len(df), os.path.basename(file)))
            for i in tqdm(range(len(df))):
                img_id = df.iat[i, 0]
                img_idx = int(img_id[img_id.find('_') + 1:])
                img = df.iloc[i, 1:].to_numpy().reshape((137, 236)).astype(np.uint8)

                if img_idx % 10000 == 0 and img_idx > 0:
                    print("Processed {} images".format(i))
                    next_folder = os.path.join(dst_dir, "{}").format(img_idx // 10000)
                    os.mkdir(next_folder)
                    curr_folder = next_folder

                file_name = os.path.join(curr_folder, "{}.png".format(img_id))
                cv.imwrite(file_name, img, [cv.IMWRITE_PNG_COMPRESSION, 0])


class BengaliTestData(Dataset):
    """
    Dataset Class (Test Time) for Bengali Parquet Data.
    """

    HEIGHT = 137
    WIDTH = 236

    def __init__(self, fp, tsfms):
        """
        Constructor
        :param fp: file path(s), if a directory was provided, will read all parquet files from the directory
        :param tsfms: image preprocessing transformers
        """

        self.tsfms = tsfms
        df = pd.read_parquet(fp)
        self.data_id = df.iloc[:, 0].values
        self.img_data = df.iloc[:, 1:].values.reshape(-1, self.HEIGHT, self.WIDTH).astype(np.uint8)

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_id = self.data_id.iloc[idx, 0]
        img = self.img_data[idx]
        for tsf in self.tsfms:
            img = tsf(img)

        return {'id': img_id, 'x': img}


class BengaliLocalDataset(Dataset):
    """
    Dataset Class for Bengali Local Data
    """

    def __init__(self, img_src, ann_src, tsfms, augments=None):
        """
        Constructor
        :param img_src: img src folder
        :param ann_src: annotation src file csv
        :param tsfms: transformation
        :param augments: augmentations
        """

        # input image size
        self.tsfms = tsfms
        self.augments = augments
        # field
        self.img_src = img_src
        self.ann = []

        # parse annotation
        with open(file=ann_src) as csv_file:
            reader = list(csv.DictReader(f=csv_file, fieldnames=['img_id', 'root', 'vowel', 'consonant', 'grapheme']))
            row = 0
            for entry in reader:
                if row > 0:
                    self.ann.append(entry)
                row += 1

    def split(self, train_ratio=0.8):
        """
        Split the dataset into two subset: train and eval
        :param train_ratio: train ratio
        :return: train eval sub set
        """

        if train_ratio <= 0 or train_ratio >= 1:
            raise Exception("Train ratio should be between 0.1 ~ 0.99!")

        train_len = int(self.__len__()*train_ratio)
        eval_len = self.__len__() - train_len
        lengths = [train_len, eval_len]
        train_set, eval_set = random_split(self, lengths)
        return {'train': train_set, 'eval': eval_set}

    def __getitem__(self, idx):
        """
        Get an image with annotations
        :param idx: idx of the item
        :return: data entry
        """

        if isinstance(idx, str):
            idx = int(idx[idx.find('_')+1:])

        # get annotation
        annotations = self.ann[idx]
        img_id = annotations['img_id']
        root = int(annotations['root'])
        vowel = int(annotations['vowel'])
        consonant = int(annotations['consonant'])
        y = torch.Tensor([root, vowel, consonant]).int()

        # gray scale
        img_idx = int(img_id[img_id.find('_')+1:])
        f_idx = int(img_idx // 1e4)
        img_p = os.path.join(self.img_src, "{}".format(f_idx), "{}.png".format(img_id))
        img = cv.imread(img_p, cv.IMREAD_GRAYSCALE)

        for tsf in self.tsfms:
            img = tsf(img)

        if self.augments is not None:
            # for augment in self.augments:
            # TODO: Refactor this to a list of augments
            img = self.augments.proc(img)

        return {"id": img_id, "x": img, "y": y}

    def __len__(self):
        return len(self.ann)

    def view_batch(self, n_row, img_ids=None):
        """
        View a random batch (12) of images
        :param n_row: number of samples = n_row * n_row
        :param img_ids: if provide a list of image id (len = n_row * n_row), show those
        :return: void
        """

        if img_ids is None:
            batch_idxes = np.random.randint(0, self.__len__(), n_row*n_row, dtype=np.int)
        elif len(img_ids) == n_row*n_row:
            batch_idxes = img_ids
        else:
            raise Exception("Length of img_ids need to be same as n_row*n_row")

        batch_processed = []
        batch_org = []

        for idx in batch_idxes:
            # get processed image
            batch_processed.append(self[idx]['x'])

            # get original image (in gray scale)
            img_id = self[idx]['id']
            img_idx = int(img_id[img_id.find('_') + 1:])
            f_idx = int(img_idx // 1e4)
            img_p = os.path.join(self.img_src, "{}".format(f_idx), "{}.png".format(img_id))
            org_img = (255 - cv.imread(img_p, cv.IMREAD_GRAYSCALE)).astype(np.float32) / 255
            batch_org.append(torch.from_numpy(org_img)[None, :, :])

        grid_pro = utils.make_grid(batch_processed, nrow=n_row)
        grid_org = utils.make_grid(batch_org, nrow=n_row)
        plt.title("Processed Image")
        plt.axis('off')
        plt.ioff()
        plt.imshow(grid_pro.numpy().transpose(1, 2, 0))
        plt.show()
        plt.title("Original Image (Reversed)")
        plt.axis('off')
        plt.ioff()
        plt.imshow(grid_org.numpy().transpose(1, 2, 0))
        plt.show()

