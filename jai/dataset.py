from torch import randperm
import torch
from torch.utils.data import Dataset
import numpy as np


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
    :return: split dataset
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

    def split(self, train_ratio=0.8, seed=None):
        """
        Split the dataset into two subset: train and eval
        :param train_ratio: train ratio
        :param seed: seed for splitting the dataset
        :return: train eval sub set
        """

        if seed is not None:
            torch.random.manual_seed(seed)
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


class Evaluator:
    """
    Class to cache the encoding and evaluation Criteria.
    The scoring operators only work for confusion matrix that
    - row idx represent the predicted class
    - col idx represent the true class
    """

    def __init__(self, names: list or str, n_classes: list or int, criteria, avg='macro', weights=None, rf=8):
        """
        Constructor. Input category order should match with the model output.
        micro:
        Calculate metrics globally by counting the total true positives, false negatives and false positives.
        macro:
        Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into
        account.
        :param names: prediction names
        :param n_classes: number of classes for the prediction
        :param criteria: criteria for computing the score
        :param avg: averaging rule
        :param weights: weights for the sum the scores of multiple predictors
        :param rf: round off
        :return: void
        """

        assert criteria in ['accuracy', 'precision', 'recall'], "criteria must be 'accuracy', 'precision' or 'recall'"
        assert avg in ['micro', 'macro'], "avg must be either 'micro' or 'macro'"
        assert0 = isinstance(names, str) and isinstance(n_classes, int) and weights is None

        if assert0:
            names = [names]
            n_classes = [n_classes]
            weights = [1]

        assert1 = isinstance(names, list) and isinstance(n_classes, list) and isinstance(weights, list) and len(
            names) == len(n_classes) == len(weights)

        assert assert1, "names, n_classes, weights do not match!"

        self.names = names
        self.n_classes = n_classes
        self.criteria = criteria
        self.avg = avg
        self.weights = weights
        self.rf = rf

    def items(self):
        """
        Provide a generator for iterating through the encoding
        :return: a generator of name: n_classes
        """
        return zip(self.names, self.n_classes)

    def compute_score(self, cm):
        """
        compute score
        :param cm: confusion matrix
        :return: requested criteria score
        """

        match = {'accuracy': self.accuracy, 'recall': self.recall, 'precision': self.precision}
        return match[self.criteria](cm)

    def accuracy(self, cm: np.ndarray):
        """
        Compute the overall accuracy score
        :param cm: confusion matrix
        :return: accuracy
        """

        return round(cm.diagonal().sum()/(cm.sum()+1e-8), self.rf)

    def precision(self, cm: np.ndarray):
        """
        Compute the average precision score
        :param cm: confusion matrix
        :return: averaged f1 score
        """

        tp = cm.diagonal()
        preds = cm.sum(axis=1)
        if self.avg == 'micro':
            return round(tp.sum()/(preds.sum() + 1e-8), 8)
        else:
            return round((tp/(preds + 1e-8)).mean(), 8)

    def recall(self, cm):
        """
        Compute the average recall score
        :param cm: confusion matrix
        :return: averaged f1 score
        """

        tp = cm.diagonal()
        acts = cm.sum(axis=0)
        if self.avg == 'micro':
            return round(tp.sum()/(acts.sum() + 1e-8), 8)
        else:
            return round((tp/(acts + 1e-8)).mean(), 8)

    def avg_over_predictor(self, pooled):
        """
        Average score across predictors
        :param pooled: pooled scores across predictors
        :return: avg scores across predictors
        """

        arr1 = np.array(pooled)
        arr2 = np.array(self.weights)
        return (arr1 * arr2).sum() / (arr2.sum() + 1e-8)

