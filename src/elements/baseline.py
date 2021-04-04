"""
Baseline mech

Author: Jia Geng
Email: gjia0214@gmail.com | jxg570@miami.edu
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple MLP network
    """

    def __init__(self, n_in_features: int, n_classes: int,  hidden_sizes: list):
        """
        MLP,
        :param n_in_features: number of input features
        :param n_classes: number of predicted classes
        :param hidden_sizes: hidden layer sizes (n_features)
        """

        super().__init__()
        modules = []
        self.n = n_in_features

        for n_out_features in hidden_sizes:
            modules.append(nn.Linear(in_features=n_in_features, out_features=n_out_features))
            modules.append(nn.ReLU(inplace=True))
            n_in_features = n_out_features

        modules.append(nn.Linear(in_features=n_in_features, out_features=n_classes))
        self.mlp = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor):
        """
        Overwrite forward pass
        :param x: input
        :return: output
        """

        return self.mlp.forward(x.view(-1, self.n))


class MLC(nn.Module):
    """
    Simple multiple conv layer networks
    """

    def __init__(self, n_in_channels: int, n_classes: int, n_intermediate_channels: list,
                 strides: list, ksize: int):
        """
        MLP,
        :param n_in_channels: number of input features
        :param n_classes: number of predicted classes
        :param n_intermediate_channels: intermediate feature channels
        :param ksize: kernel size
        """

        super().__init__()
        modules = []

        if len(n_intermediate_channels) != len(strides):
            raise Exception("Length of n_intermediate_channels must match with length of strides")

        for n_out_channels, stride in zip(n_intermediate_channels, strides):
            conv = nn.Conv2d(in_channels=n_in_channels, out_channels=n_out_channels,
                             kernel_size=ksize, stride=stride, padding=ksize//2)
            bn = nn.BatchNorm2d(n_out_channels)
            relu = nn.ReLU(inplace=True)
            modules.append(nn.Sequential(conv, bn, relu))
            n_in_channels = n_out_channels

        modules.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.mlc = nn.Sequential(*modules)
        self.fc = nn.Linear(in_features=n_in_channels, out_features=n_classes)

    def forward(self, x: torch.Tensor):
        """
        Overwrite forward pass
        :param x: input
        :return: output
        """

        x = self.mlc(x)
        x = self.fc(x.view(x.shape[0], -1))
        return x
