"""
Head Module
Convert the final feature maps to class scores

Author: Jia Geng
Email: gjia0214@gmail.com | jxg570@miami.edu
"""

import torch.nn as nn
import torch


class AvgPoolFCHead(nn.Module):
    """
    Average Pooling + Dense Layers
    avgpool - fc1 - relu - fc2 - scores
    """

    def __init__(self, n_classes: int, in_channels: int, buffer_reduction: int or None, avgpool_target_shape: tuple or int):
        """
        Constructor
        :param n_classes: number of classes
        :param in_channels: number of feature maps
        :param buffer_reduction: if None, use a single FC layer to class score, otherwise, use FC-relu-FC
        :param avgpool_target_shape: the target shape for the avgpool layer output. Default is (1, 1), i.e. each feature map result in one 
        flattened feature. For (4, 4), each feature map will result in 4x4=16 flattened features.
        """

        super().__init__()

        # average pool
        self.avgpool = nn.AdaptiveAvgPool2d(avgpool_target_shape)
        ratio = avgpool_target_shape[0] * avgpool_target_shape[1] if isinstance(avgpool_target_shape, tuple) else avgpool_target_shape
        n_flatten_features = in_channels * ratio

        # FCs
        if buffer_reduction is not None:
            buffer_size = n_flatten_features // buffer_reduction
            fc1 = nn.Linear(n_flatten_features, buffer_size)
            relu = nn.ReLU(inplace=True)
            fc2 = nn.Linear(buffer_size, n_classes)
            fcs = [fc1, relu, fc2]
        else:
            fcs = [nn.Linear(n_flatten_features, n_classes)]

        # wrap together
        self.fcs = nn.Sequential(*fcs)

    def forward(self, x):
        x = self.fcs(self.avgpool(x).view(x.shape[0], -1))
        return x

