"""
Building blocks for the ConvNet mech

Author: Jia Geng
Email: gjia0214@gmail.com | jxg570@miami.edu
"""

import torch
import torch.nn as nn
from torch import Tensor


def conv2d(num_in_channels: int, num_out_channels: int, ksize: int, stride: int, bias: bool, padding='maintain', separable_convolution=False):
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
    :param separable_convolution: whether to use the depth-wise separable covolution
    :return: 2D conv layer
    """

    if padding == 'maintain':
        padding = ksize // 2
    else:
        assert isinstance(padding, int), 'padding must be either maintain or a integer'

    if not separable_convolution:
        conv = nn.Conv2d(in_channels=num_in_channels, out_channels=num_out_channels,
                         kernel_size=ksize, stride=stride, padding=padding, bias=bias)
    else:
        conv = DepthwiseSeparableConv(in_channels=num_in_channels, out_channels=num_out_channels,
                                      kernel_size=ksize, stride=stride, bias=bias)

    return conv


def conv_relu_bn(num_in_channels: int, num_out_channels: int, ksize: int, stride: int, bias: bool,
                 separable_convolution=False):
    """
    conv-relu-bn unit
    :param num_in_channels: number of input channels
    :param num_out_channels: number of output channels
    :param ksize: kernel size
    :param stride: stride
    :param bias: whether to use bias
    :param separable_convolution: whether to use separable_convolution
    :return: conv-relu-bn unit
    """

    conv = conv2d(num_in_channels=num_in_channels, num_out_channels=num_out_channels,
                  ksize=ksize, stride=stride, bias=bias,
                  separable_convolution=separable_convolution)
    bn = nn.BatchNorm2d(num_out_channels)
    relu = nn.ReLU(inplace=True)

    return nn.Sequential(conv, bn, relu)


class DepthwiseSeparableConv(nn.Module):
    """
    Depth-wise separable convolution from MobileNet
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, bias: bool):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: stride
        :param bias: whether to use bias
        """

        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=kernel_size // 2, groups=in_channels, stride=stride, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out


class InitConvBlock(nn.Module):
    """
    Initial con-relu-bn-maxpool unit
    """

    def __init__(self, num_out_channels: int, conv_stride: int, conv_ksize: int,
                 pool_stride: int, pool_ksize: int,
                 num_in_channels=3, separable_convolution=False):
        """
        Constructor
        :param num_out_channels: number of output channels
        :param conv_stride: stride for the conv layer
        :param conv_ksize: kernel size for the conv layer
        :param pool_stride: stride for the maxpool layer
        :param pool_ksize: kernel size for the maxpool layer
        :param num_in_channels: number of input channel, default is 3
        :param separable_convolution: whether to use separable_convolution
        """

        super().__init__()

        # build layers
        self.init_convrelubn = conv_relu_bn(num_in_channels=num_in_channels, num_out_channels=num_out_channels,
                                            stride=conv_stride, ksize=conv_ksize, bias=False,
                                            separable_convolution=separable_convolution)
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

    def __init__(self, num_units: int, num_in_channels: int, ksize, space_reduction_ratio=2, feature_expand_ratio=2, num_layer_per_unit=2,
                 separable_convolution=False):
        """
        Constructor
        :param num_units: number of residual units, each residual unit contains two (or more) conv-relu-bn layers and a shortcut connection
        :param num_in_channels: number of input feature maps
        :param ksize: kernel size
        :param space_reduction_ratio: stride for the first conv-relu-bn unit in the first res layer, default is 2
        :param feature_expand_ratio: feature expand ratio for the first conv-relu-bn unit in the first res layer, default is 2
        :param num_layer_per_unit: number of conv-relu-bn layer in each res unit, default is 2
        :param separable_convolution: whether to use separable_convolution
        """
        super().__init__()

        # layer name
        self.names = ['res_layer{}'.format(i+1) for i in range(num_units)]

        # build the block
        n_features = feature_expand_ratio * num_in_channels
        for i, name in enumerate(self.names):
            if i == 0:
                # For the first conv-relu-bn unit in the first res layer, downsample the space and increase the feature map
                res_layer = _ResidualUnit(num_in_channels=num_in_channels, init_conv_stride=space_reduction_ratio,
                                          feature_expand_ratio=feature_expand_ratio, ksize=ksize, num_layers=num_layer_per_unit,
                                          separable_convolution=separable_convolution)
            else:
                # No change on num. of feature maps or space
                res_layer = _ResidualUnit(num_in_channels=n_features, init_conv_stride=1,
                                          feature_expand_ratio=1, ksize=ksize, num_layers=num_layer_per_unit,
                                          separable_convolution=separable_convolution)

            # pytorch overwrote the __setattr__ method
            self.add_module(name=name, module=res_layer)

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

    def __init__(self, num_in_channels: int, init_conv_stride: int, feature_expand_ratio: int, ksize: int, num_layers=2,
                 separable_convolution=False):
        """
        Constructor
        :param num_in_channels: number of input channels
        :param init_conv_stride: stride for initial conv layer, affect output H, W
        :param feature_expand_ratio: feature expansion ratio, affect C
        :param ksize: kernel size K
        :param num_layers: number of conv-relu-bn units, default is 2
        :param separable_convolution: whether to use separable_convolution
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
                                     stride=init_conv_stride, ksize=ksize, bias=False,
                                     separable_convolution=separable_convolution)
            else:
                layer = conv_relu_bn(num_in_channels=n_features, num_out_channels=n_features,
                                     stride=1, ksize=ksize, bias=False,
                                     separable_convolution=separable_convolution)
            self.add_module(name=self.names[i], module=layer)

        # add shortcut connection
        self.add_module(name='shortcut',
                        module=_ShortcutConnection(num_in_channel=num_in_channels, num_out_channel=n_features, stride=init_conv_stride))

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
        :param stride: stride for matching the feature map size (H, W) before and after
        """

        super().__init__()
        self.bottleneck = None

        # if channel does not match or space does not match, add a bottle neck
        if num_in_channel != num_out_channel or stride != 1:
            self.bottleneck = nn.Sequential(conv2d(num_in_channel, num_out_channel, ksize=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(num_out_channel))

    def forward(self, x: Tensor, identity: Tensor):
        """
        Forward pass
        :param x: output from the previous layer
        :param identity: identity tensor that will be add to x
        :return:
        """

        if self.bottleneck is None:
            return x + identity
        else:
            return x + self.bottleneck(identity)


class DenseBlock(nn.Module):
    """
    DenseNet basic block, consist of a stack of dense unit
    Each dense unit is a stack of bottelneck + convrelubn layer with dense forward pass
    Input (H, W, C_in)
    Output (H, W, C_in + growth_rate * num_units)
    Each dense layer:
    - consist of a bottleneck layer and a conv layer
        - bottleneck layer will reduce the num. of feature maps to base size * growth rate
        - conv layer will reduce the space size, it will also reduce the feature maps to growth rate
    - take in concat of all previous learned feature maps and return k (growth rate) feature maps
    """

    def __init__(self, num_units: int, num_in_channels: int, growth_rate: int,
                 base_bottleneck_size: int, ksize: int, dropout_rate: float,
                 separable_convolution=False):
        """
        :param num_units: number of dense layers
        :param num_in_channels: number of input channel
        :param growth_rate: how many filters to be added after each layer (`k` in paper)
        :param base_bottleneck_size: multiplicative factor for number of bottle neck layers (i.e. bn_size * k features in the bottleneck layer)
        :param ksize: kernel size
        :param dropout_rate: dropout rate after each dense layer
        :param separable_convolution: whether to use separable convolution
        """
        super().__init__()

        # layer names
        self.names = ['dense_layer{}'.format(i+1) for i in range(num_units)]

        # build the block
        for i, name in enumerate(self.names):
            # num_in_channel is the concatenate of all previous layers' output & original input
            # each layer's output channel = growth_rate, hence, for i-th layer
            # num_in_channels + i * growth_rate
            layer = _DenseUnit(num_in_channels=num_in_channels + i * growth_rate,
                               growth_rate=growth_rate, base_bottleneck_size=base_bottleneck_size,
                               ksize=ksize, drop_rate=dropout_rate, separable_convolution=separable_convolution)

            self.add_module(name=name, module=layer)

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


class _DenseUnit(nn.Module):
    """
    Dense layer consists of:
    - At bottleneck, the channels will be mapped to bn_size * growth rate
    - At the conv-relu-bn unit, the channels will be reduced to growth rate
    - Name growth rate because each layer's output will be concat to the input of following layers.
    """

    def __init__(self, num_in_channels: int, growth_rate: int, base_bottleneck_size: int, ksize: int, drop_rate: float,
                 separable_convolution=False):
        """
        :param num_in_channels: number of input channel
        :param growth_rate: how many filters to add each layer (`k` in paper)
        :param base_bottleneck_size: multiplicative factor that decides the intermediate num. of feature maps (= base_bottleneck_size * k)
        :param ksize: kernel size
        :param drop_rate: dropout rate after each dense layer
        :param separable_convolution: whether to use separable_convolution
        """

        super().__init__()

        # dense unit start with a bottle neck layer to limit the over growth of the feature map
        self.bottleneck = conv_relu_bn(num_in_channels=num_in_channels, num_out_channels=base_bottleneck_size * growth_rate,
                                       ksize=1, stride=1, bias=False)
        self.convrelubn = conv_relu_bn(num_in_channels=base_bottleneck_size * growth_rate, num_out_channels=growth_rate,
                                       ksize=ksize, stride=1, bias=False, separable_convolution=separable_convolution)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x: Tensor):
        return self.dropout(self.convrelubn(self.bottleneck(x)))


class TransitionBlock(nn.Module):
    """
    Transition Block for DenseNet
    conv-relu-bn-avgpool
    Use between DenseBlocks to reduce the number of feature maps and the space size.
    """

    def __init__(self, num_in_channels, feature_reduction_ratio=2, space_reduction_ratio=2, avgpool_ksize=2):
        """
        Constructor
        :param num_in_channels: num. of input channels
        :param feature_reduction_ratio: reduction ratio for feature maps, num. of output feature maps = num_in_channels // reduction ratio
        :param space_reduction_ratio: reduction ratio for space, H_out, W_out = H_in // ratio, W_in // ratio
        """

        super().__init__()

        # bottleneck unit to reduce the number
        num_out_channels = num_in_channels // feature_reduction_ratio
        self.bottleneck = conv_relu_bn(num_in_channels=num_in_channels, num_out_channels=num_out_channels, ksize=1, stride=1, bias=False)

        # TODO: whether padding is needed here
        # avg pooling to reduce the space (H-1//2, W-1//2)
        self.avgpool = nn.AvgPool2d(kernel_size=avgpool_ksize, stride=space_reduction_ratio)

    def forward(self, x: Tensor):
        """
        Forward pass
        :param x: input
        :return: output with less feature maps and space
        """

        return self.avgpool.forward(self.bottleneck.forward(x))

