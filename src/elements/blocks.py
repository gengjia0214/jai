"""
Building blocks for the ConvNet models

Author: Jia Geng
Email: gjia0214@gmail.com | jxg570@miami.edu
"""

import torch
import torch.nn as nn
from torch import Tensor


def conv2d(num_in_channels: int, num_out_channels: int, ksize: int, stride: int, bias: bool, padding='maintain'):
    """
    2D Conv layer
    Input: (H, W, C_in)
    Kernel size: K
    Stride: S
    Padding: P
    Output: ( (H-K//2+P)//S, (W-K//2+P)//S, C_out )
    Kernel size: (K, K)
    Padding: default is to pad with K//2 so that no H, W loss caused by the sliding
    :param num_in_channels: number of input channels C_in
    :param num_out_channels: number of output channels C_out
    :param ksize: kernel size (K, K)
    :param stride: stride for the convolution operations
    :param bias: whether to use bias
    :param padding: padding for the conv layer, default is ksize//2 to maintain the output
    :return: 2D conv layer
    """

    if padding == 'maintain':
        padding = ksize // 2
    else:
        assert isinstance(padding, int), 'padding must be either maintain or a integer'

    return nn.Conv2d(in_channels=num_in_channels, out_channels=num_out_channels,
                     kernel_size=ksize, stride=stride, padding=padding, bias=bias)


def conv_relu_bn(num_in_channels: int, num_out_channels: int, ksize: int, stride: int, bias: bool):
    """
    conv-relu-bn unit
    :param num_in_channels: number of input channels
    :param num_out_channels: number of output channels
    :param ksize: kernel size
    :param stride: stride
    :param bias: whether to use bias
    :return: conv-relu-bn unit
    """

    conv = conv2d(num_in_channels=num_in_channels, num_out_channels=num_out_channels, ksize=ksize, stride=stride, bias=bias)
    bn = nn.BatchNorm2d(num_out_channels)
    relu = nn.ReLU(inplace=True)

    return nn.Sequential(conv, bn, relu)


class InitConvBlock(nn.Module):
    """
    Initial con-relu-bn-maxpool unit
    """

    def __init__(self, num_out_channels: int, conv_stride: int, conv_ksize: int,
                 pool_stride: int, pool_ksize: int,
                 num_in_channels=3):
        """
        Constructor
        :param num_out_channels: number of output channels
        :param conv_stride: stride for the conv layer
        :param conv_ksize: kernel size for the conv layer
        :param pool_stride: stride for the maxpool layer
        :param pool_ksize: kernel size for the maxpool layer
        :param num_in_channels: number of input channel, default is 3
        """

        super().__init__()

        # build layers
        self.init_convrelubn = conv_relu_bn(num_in_channels=num_in_channels, num_out_channels=num_out_channels,
                                            stride=conv_stride, ksize=conv_ksize, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=pool_ksize, stride=pool_stride, padding=pool_ksize // 2)

    def forward(self, x: Tensor):
        """
        Forward pass
        :return:
        """

        return self.maxpool.forward(self.init_convrelubn.forward(x))


class ResidualBlock(nn.Module):
    """
    Residual Block
    A stack of residual layers with same num. feature maps
    Each residual layer contains two conv-relu-bn unit and a shortcut connection
    For the first conv-relu-bn unit in the first res layer, downsample the space and increase the feature map
    Input: (H, W, C_in)
    Output: (H // init_conv_stride, W // init_conv_stride, C_in * feature_expand_ratio )
    """

    def __init__(self, num_units: int, num_in_channels: int, ksize, init_conv_stride=2, feature_expand_ratio=2, num_layer_per_unit=2):
        """
        Constructor
        :param num_units: number of residual units, each residual unit contains two (or more) conv-relu-bn layers and a shortcut connection
        :param num_in_channels: number of input feature maps
        :param ksize: kernel size
        :param init_conv_stride: stride for the first conv-relu-bn unit in the first res layer, default is 2
        :param feature_expand_ratio: feature expand ratio for the first conv-relu-bn unit in the first res layer, default is 2
        :param num_layer_per_unit: number of conv-relu-bn layer in each res unit, default is 2
        """
        super().__init__()

        # layer name
        self.names = ['res_layer{}'.format(i+1) for i in range(num_units)]

        # build the block
        n_features = feature_expand_ratio * num_in_channels
        for i, name in enumerate(self.names):
            if i == 0:
                # For the first conv-relu-bn unit in the first res layer, downsample the space and increase the feature map
                res_layer = _ResidualUnit(num_in_channels=num_in_channels, init_conv_stride=init_conv_stride,
                                          feature_expand_ratio=feature_expand_ratio, ksize=ksize, num_layers=num_layer_per_unit)
            else:
                # No change on num. of feature maps or space
                res_layer = _ResidualUnit(num_in_channels=n_features, init_conv_stride=1,
                                          feature_expand_ratio=1, ksize=ksize, num_layers=num_layer_per_unit)

            # pytorch overwrote the __setattr__ method
            self.__setattr__(name=name, value=res_layer)

    def forward(self, x: Tensor):
        """
        Forward pass
        :param x: input
        :return: output (H // init_conv_stride, W // init_conv_stride, C_in * feature_expand_ratio )
        """

        for name in self.names:
            x = self._modules[name].forward(x)

        return x


class _ResidualUnit(nn.Module):
    """
    Basic resnet layer
    Stacks of conv-relu-bn units with skip connection every two conv-relu-bn unit
    Stride and the feature expanding is executed in the first conv-relu-bn unit
    Padding is used to prevent space loss caused by sliding
    Input: (H, W, C)
    Output: (H // S, W // S, C*expand_ratio)
    """

    def __init__(self, num_in_channels: int, init_conv_stride: int, feature_expand_ratio: int, ksize: int, num_layers=2):
        """
        Constructor
        :param num_in_channels: number of input channels
        :param init_conv_stride: stride for initial conv layer, affect output H, W
        :param feature_expand_ratio: feature expansion ratio, affect C
        :param ksize: kernel size K
        :param num_layers: number of conv-relu-bn units, default is 2
        """

        super().__init__()

        # unit names
        self.names = ['convrelubn{}'.format(i+1) for i in range(num_layers)]
        # number of output feature maps
        n_features = num_in_channels * feature_expand_ratio

        # build layers
        # stack conv-relu-bn units
        for i in range(num_layers):
            if i == 0:  # only the first layer will apply the reduction in space and/or expansion in features
                layer = conv_relu_bn(num_in_channels=num_in_channels, num_out_channels=n_features,
                                     stride=init_conv_stride, ksize=ksize, bias=False)
            else:
                layer = conv_relu_bn(num_in_channels=n_features, num_out_channels=n_features,
                                     stride=1, ksize=ksize, bias=False)
            self.__setattr__(name=self.names[i], value=layer)

        # add shortcut connection
        self.__setattr__(name='shortcut',
                         value=_ShortcutConnection(num_in_channel=num_in_channels, num_out_channel=n_features, stride=init_conv_stride))

    def forward(self, x: Tensor):
        """
        Forward pass
        :param x: input tensor
        :return: feature maps
        """

        # identity
        identity = x
        # go through conv-relu-bn units
        for layer_name in self.names:
            x = self._modules[layer_name].forward(x)
        # shortcut connection
        x = self._modules['shortcut'].forward(x, identity)

        return x


class _ShortcutConnection(nn.Module):
    """
    Skip connection
    Use 1x1 conv to match the dimension of connected feature maps
    """

    def __init__(self, num_in_channel: int, num_out_channel: int, stride: int):
        """
        Constructor
        :param num_in_channel: number of input channels
        :param num_out_channel: number of output channels
        :param stride: stride for matching the H, W
        """

        super().__init__()
        self.mapping = None

        # if channel does not match or space does not match, add a bottle neck
        if num_in_channel != num_out_channel or stride != 1:
            self.mapping = conv2d(num_in_channel, num_out_channel, ksize=1, stride=stride, bias=False)

    def forward(self, x: Tensor, identity: Tensor):
        """
        Forward pass
        :param x: output from the previous layer
        :param identity: identity tensor that will be add to x
        :return:
        """

        if self.mapping is None:
            return x + identity
        else:
            return x + self.mapping(identity)


class DenseBlock(nn.Module):
    """
    DenseNet basic block, consist of a stack of dense layers
    Input (H, W, C_in)
    Output (H, W, C_in + growth_rate * num_layers)
    Each dense layer:
    - consist of a bottle neck layer and a conv layer
    - take in concat of all previous learned feature maps and return k (growth rate) feature maps
    """

    def __init__(self, num_layers: int, num_in_channels: int, growth_rate: int, bn_size: int, ksize: int, drop_rate: float):
        """
        :param num_layers: number of dense layers
        :param num_in_channels: number of input channel
        :param growth_rate: how many filters to add each layer (`k` in paper)
        :param bn_size: multiplicative factor for number of bottle neck layers (i.e. bn_size * k features in the bottleneck layer)
        :param ksize: kernel size
        :param drop_rate: dropout rate after each dense layer
        """
        super().__init__()

        # layer names
        self.names = ['dense-layer{}'.format(i+1) for i in range(num_layers)]

        # build the block
        for i, name in enumerate(self.names):
            # num_in_channel is the concatenate of all previous layers' output & original input
            # each layer's output channel = growth_rate, hence, for i-th layer
            # num_in_channels + i * growth_rate
            layer = self.dense_layer(num_in_channels=num_in_channels + i * growth_rate,
                                     growth_rate=growth_rate, bn_size=bn_size, ksize=ksize, drop_rate=drop_rate)
            self.__setattr__(name=name, value=layer)

    def forward(self, x: Tensor):
        """
        Forward pass
        :param x: input (H, W, C_in)
        :return: output (H, W, C_in + n_layers*growth_rate), concatenation of input, output from each layer
        """

        # list containing all the features returned by each layer
        features = [x]

        # iteration, at each step, concat the current features as the input and collect new feature into the list
        for name in self.names:
            concat_features = torch.cat(features, dim=1)
            new_features = self._modules[name].forward(concat_features)
            features.append(new_features)

        # return concat features
        return torch.cat(features, dim=1)

    @staticmethod
    def dense_layer(num_in_channels: int, growth_rate: int, bn_size: int, ksize: int, drop_rate: float):
        """
        Get a dense unit: bottelneck -> conv-relu-bn
        - At bottleneck, the channels will be mapped to bn_size * growth rate
        - At the conv-relu-bn unit, the channels will be reduced to growth rate
        - Name growth rate because each layer's output will be concat to the input of following layers.
        :param num_in_channels: number of input channel
        :param growth_rate: how many filters to add each layer (`k` in paper)
        :param bn_size: multiplicative factor for number of bottle neck layers (i.e. bn_size * k features in the bottleneck layer)
        :param ksize: kernel size
        :param drop_rate: dropout rate after each dense layer
        """

        # dense unit start with a bottle neck layer to limit the growth of the
        bottleneck = conv_relu_bn(num_in_channels=num_in_channels, num_out_channels=bn_size*growth_rate, ksize=1, stride=1, bias=False)
        convrelubn = conv_relu_bn(num_in_channels=bn_size*growth_rate, num_out_channels=growth_rate, ksize=ksize, stride=1, bias=False)
        dropout = nn.Dropout(p=drop_rate)

        return nn.Sequential(bottleneck, convrelubn, dropout)


class TransitionBlock(nn.Module):
    """
    Transition Block for DenseNet
    conv-relu-bn-avgpool
    Use between DenseBlocks to reduce the number of feature maps and the space size.
    """

    def __init__(self, num_in_channels, num_out_channels):
        """
        Constructor
        :param num_in_channels: num. of input channels
        :param num_out_channels: num. of output channels
        """
        super().__init__()

        # bottleneck unit to reduce the number
        self.bottleneck = conv_relu_bn(num_in_channels=num_in_channels, num_out_channels=num_out_channels, ksize=1, stride=1, bias=False)
        # avg pooling to reduce the space (H-1//2, W-1//2)
        # TODO: whether padding is needed here
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor):
        """
        Forward pass
        :param x: input
        :return: output with less feature maps and space
        """

        return self.avgpool.forward(self.bottleneck.forward(x))


class InvertedResidualBlock(nn.Module):
    """
    TODO: implement the mobilenet base block
    """
    pass
