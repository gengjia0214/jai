from torch import randperm
from torch.utils.data import Dataset
import torch


def _accumulate(iterable, fn=lambda x, y: x + y):
    """Taken from PyTorch"""

    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


def jai_split(dataset, lengths):
    """
    Random split for JaiDataset
    :param dataset: input dataset
    :param lengths: lengths
    :return: splitted dataset
    """

    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths)).tolist()
    return [JaiSubset(dataset, indices[offset - length:offset], dataset.augmentators) for offset, length in zip(
        _accumulate(lengths), lengths)]


class JaiDataset(Dataset):
    """
    TODO: refactor the id access and string ID access to this parent class
    Abstract class for Jai Dataset that supports augmentation
    """

    def __init__(self, tsfms, augments=None, *args):
        self.tsfms = tsfms
        self.augmentators = augments

    def __getitem__(self, item):
        raise NotImplemented("")

    def __len__(self):
        raise NotImplemented("")

    def prepro(self, img):
        """
        Preprocess the data
        :param img: image
        :return: void
        """

        for tsf in self.tsfms:
            img = tsf(img)

        return img

    def augment(self, img):
        """
        Don't forget to call this before getitem.
        :param img:
        :return:
        """

        if self.augmentators is not None:
            # For augment in self.augments:

            if isinstance(self.augmentators, list):
                for aug in self.augmentators:
                    img = aug.proc(img)
                return img
            else:
                return self.augmentators.proc(img)

    def split(self, train_ratio=0.8):
        """
        Split the dataset into two subset: train and eval
        :param train_ratio: train ratio
        :return: train eval sub set
        """

        if train_ratio <= 0 or train_ratio >= 1:
            raise Exception("Train ratio should be between 0 ~ 1!")

        train_len = int(self.__len__()*train_ratio)
        eval_len = self.__len__() - train_len
        lengths = [train_len, eval_len]
        train_set, eval_set = jai_split(self, lengths)
        return {'train': train_set, 'eval': eval_set}


class JaiSubset(JaiDataset):
    """
    Jai-style Subset
    TODO: refactor the string ID access to make it compatible with parent class
    """

    def __init__(self, dataset, indices, augments=None):
        super().__init__(augments)
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class DataClassDict:
    """
    Dummy class to cache the encoding
    """

    def __init__(self, names: list or str, n_classes: list or int):
        """
        Constructor. Input category order should match with the model output.
        :param names: prediction names
        :param n_classes: number of classes for the prediction
        :return: void
        """

        assert0 = isinstance(names, str) and isinstance(n_classes, int)
        if assert0:
            names = [names]
            n_classes = [n_classes]
        assert1 = isinstance(names, list) and isinstance(n_classes, list) and len(names) == len(n_classes)

        assert assert1, "names and n_classes do not match!"

        self.names = names
        self.n_classes = n_classes

    def items(self):
        return zip(self.names, self.n_classes)

