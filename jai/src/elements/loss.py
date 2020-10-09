"""
Loss Function Module

Author: Jia Geng
Email: gjia0214@gmail.com | jxg570@miami.edu
"""

import torch
import torch.nn as nn
from torch import Tensor


class WeightedCrossEntropy(nn.Module):
    """
    Weighted cross entropy loss
    """

    def __init__(self, n_classes: int, weights=None):
        """
        Constructor
        :param n_classes: number of classes (label should be 0, 1, 2, ..., n_classes-1)
        :param weights: loss weights for each class, if none, then equal weighted (1:1:1...), or
        """

        super().__init__()

        # sanity check
        if weights is None:
            weights = [1] * n_classes
        elif len(weights) != n_classes:
            raise Exception('Weight length {} does not match with number of classes {}'.format(len(weights), n_classes))

        # flag for device TODO: find a better way to handle the device management
        self.on_device = False
        self.loss_module = nn.CrossEntropyLoss(weight=torch.Tensor(weights))

    def forward(self, X: Tensor, Y: Tensor):
        """
        Forward Pass
        :param X: input
        :param Y: target
        :return: loss in Tensor
        """

        if not self.on_device:
            self.loss_module.weight = self.loss_module.weight.to(X.device)
            self.on_device = True

        loss = self.loss_module.forward(input=X, target=Y)

        return loss

