"""
Data handling classes

Author: Jia Geng
Email: gjia0214@gmail.com | jxg570@miami.edu
"""

import os
from PIL import Image
from torchvision.transforms.transforms import *
from torch.utils.data.dataloader import *
from torch.utils.data.dataset import *


def default_preprocessing(mean, std, size: tuple):
    """
    Get the default preprocessing pipeline
    ToPIL -> Resize -> ToTensor -> Normalization
    :return: list of processing module
    """

    return [ToPILImage(), Resize(size=size), ToTensor(), Normalize(mean=mean, std=std)]


def default_augmentation(p):
    """
    Get the default augmentation.
    Random Horizontal Flip & RandomVerticalFlip & Random Jitter
    :return: list of augmentation techniques
    """

    # 0.5 chance of applying the flipping
    # 0.5 chance of horizontal flip & 0.5 chance of vertical flip
    # overall 0.5 x 1.0 + 0.5 x 0.25 = 0.625 chance of Not get flipped
    flip = RandomApply([RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5)], p)

    # 0.25 chance of get color jittered (disable the jitter on hue as it would require PIL input)
    jitter = RandomApply([ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0)], p=0.25)

    return [flip, jitter]


class DataPackerAbstract:
    """
    An abstract class for packing the dataset.
    The abstract will make sure the packed data container class will be compatible with the ProcDataset
    """

    def __init__(self):

        self.mode = None  # mode
        self.data_src = None  # data src should contains id & image data or image fp
        self.labels = None  # labels
        self.annotations = None  # other annotations for the data, should contains id & annotations if provided

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
        - annotations should be load in memory
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
        if self.annotations is not None: assert isinstance(self.labels, dict), 'Annotation should be a dictionary with key: id'
        assert self.mode in ['disk', 'memory'], 'model must be either disk or memory'

        for data_id in self.data_src:
            if self.annotations is not None and data_id not in self.annotations:
                raise Exception('Data id={} from data src can not be found in annotation.'.format(data_id))

        output = {'mode': self.mode, 'data': self.data_src, 'labels': self.labels}
        if self.annotations is not None:
            output['annotations'] = self.annotations

        return output

    def split(self, *args):
        """
        Split the packed data into train and evaluation, test
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

    def __init__(self, data_packer: DataPackerAbstract, pre_pipeline: list or None, augmentations: list or None, device='cpu', data_src_dir=None):
        """
        Constructor
        :param data_packer: A packed data object. The object class should inherit the PackedDataAbstract
        :param pre_pipeline: pre-processing pipeline, apply to image for both training and testing time. Note that ToTensor is already included,
        no need to pass it again.
        :param augmentations: Data augmentations. If not None, pass a list of augmentation module. At training time, each image will be
        augmented by the provided augmentation technique.
        :param device: device for computation (preprocessing & augmentation)
        :param data_src_dir: the directory for the data, useful when mode is disk, if None, treat the path provided as full path
        """

        # phase
        self.training = False

        # data
        self.packed_data = data_packer.get_packed_data()
        self.parent_dir = data_src_dir
        self.device = device

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
        :param idx:
        :return: data and annotations
        """

        # get the image id
        data_id = self.idx2id[idx]

        # get the mode
        mode = self.packed_data['mode']

        # get he image
        if mode == 'disk':
            base_path = self.packed_data['data'][data_id]
            fp = os.path.join(self.parent_dir, base_path) if self.parent_dir is not None else base_path
            img = Image.open(fp=fp)
        elif mode == 'memory':
            img = self.packed_data['data'][data_id]
        else:
            raise Exception('Mode must be either disk or memory but was {}'.format(mode))

        # convert the image to tensor
        img = ToTensor()(img).to(self.device)

        # apply preprocessing pipeline
        if self.pre_processing is not None:
            img = self.pre_processing(img)

        # apply augmentations
        if self.training and self.augmentations is not None:
            img = self.augmentations(img)

        # get the labels
        label = self.packed_data['labels'][data_id]

        output = {'x': img, 'y': label}

        # get the additional annotations, if any
        if 'annotations' in self.packed_data:
            output['annotations'] = self.packed_data['annotations'][data_id]

        # TODO: need to test the data mini-batching for annotation. Might need to return label and other useful annotation such as area etc
        #  respectively. Add some more later
        return output

    def __len__(self):
        """
        Get the length of the data
        :return: length of the data
        """

        return len(self.idx2id)

    def train(self):
        """
        Switch to train phase
        :return: void
        """

        self.training = True

    def eval(self):
        """
        Switch to eval phase
        :return: void
        """

        self.training = False

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

    def __init__(self, train_dataset: ImgDataset or None, eval_dataset: ImgDataset or None, batch_size):
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
