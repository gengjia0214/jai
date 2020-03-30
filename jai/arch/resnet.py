from collections import OrderedDict
from torch import Tensor
import torch
import torch.nn as nn


"""
input image 49x49x3

input       layer       ksize   stride  pad     filter  ratio       Note

49x49x3     conv        5x5     1       2       32      75->32      no stride to prevent info loss
49x49x32    bn          NA      NA      NA      NA      NA          
49x49x32    relu        NA      NA      NA      NA      NA
49x49x32    maxpool     2x2     1       1       NA      4->1        no stride to prevent info loss

49x49x32    conv        3x3     1       1       64      9->2        still working with original size                   
49x49x64    bn          NA      NA      NA      NA      NA
49x49x64    relu        NA      NA      NA      NA      NA
49x49x64    conv        3x3     1       1       64      9->1                         
49x49x64    bn          NA      NA      NA      NA      NA
49x49x64    skipconn    NA      NA      NA      NA      NA
49x49x64    relu        NA      NA      NA      NA      NA

49x49x64    conv        3x3     2       1       128     9->2        reduce size here as features are well constructed                  
24x24x128   bn          NA      NA      NA      NA      NA
24x24x128   relu        NA      NA      NA      NA      NA
24x24x128   conv        3x3     1       1       128     9->1                         
24x24x128   bn          NA      NA      NA      NA      NA
24x24x128   skipconn    NA      NA      NA      NA      NA
24x24x128   relu        NA      NA      NA      NA      NA

24x24x128   conv        3x3     2       1       256     9->2                         
12x12x256   bn          NA      NA      NA      NA      NA
12x12x256   relu        NA      NA      NA      NA      NA
12x12x256   conv        3x3     1       1       256     9->1                         
12x12x256   bn          NA      NA      NA      NA      NA
12x12x256   skipconn    NA      NA      NA      NA      NA
12x12x256   relu        NA      NA      NA      NA      NA

12x12x256   conv        3x3     2       1       512     9->2                         
6x6x512     bn          NA      NA      NA      NA      NA
6x6x512     relu        NA      NA      NA      NA      NA
6x6x512     conv        3x3     1       1       512     9->1                         
6x6x512     bn          NA      NA      NA      NA      NA
6x6x512     skipconn    NA      NA      NA      NA      NA
6x6x512     relu        NA      NA      NA      NA      NA

6x6x512     avgpool     6x6     NA      NA      NA      36->1       avgpool indicating importance of a feature
flatten     
512         FC          512X4  NA      NA      NA       128->1        

softmax
"""


def conv(in_channels, out_channels, ksize=3, stride=1):
    """simple convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=ksize//2)


def get_settings(mode, n_classes):

    settings = OrderedDict()

    if mode == 'preserve':
        # block_name, in_channels, out_channels, conv_size, conv_stride, pool_size, pool_stride
        settings['start'] = [StartBlock, 'start_block', 3, 32, 5, 1, 2, 1]
        # block_name, n_stacks, in_channels, space_reduce, feature_expand, mode
        settings['block1'] = [BasicBlock, 'block1', 2, 32, 1, 2, 'original']
        settings['block2'] = [BasicBlock, 'block2', 2, 64, 2, 2, 'original']
        settings['block3'] = [BasicBlock, 'block3', 2, 128, 2, 2, 'original']
        settings['block4'] = [BasicBlock, 'block4', 2, 256, 2, 2, 'original']
        # block_name, in_channels, buffer_reduction, n_classes
        settings['head'] = [AvgPoolHead, 'head', 512, n_classes]
        return settings


class StartBlock(nn.Module):
    """
    The first couple of layers before go into the basic block
    conv - bn - relu  - maxpool
    """

    def __init__(self, block_name, in_channels, out_channels, conv_size, conv_stride, pool_size, pool_stride):

        super().__init__()
        self.name = block_name
        conv0 = conv(in_channels, out_channels, conv_size, conv_stride)
        bn = nn.BatchNorm2d(out_channels)  # takes in the n_features as input to set up learnable params
        relu = nn.ReLU(inplace=True)   # relu will modify con inplace
        maxpool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride, padding=pool_size//2)

        self.__setattr__(block_name, nn.Sequential(conv0, bn, relu, maxpool))

    def forward(self, x):

        for layer in self.__getattr__(self.name):
            x = layer(x)
        return x


class SkipConnection(nn.Module):
    """
    Skip connection with channel and spatial matching using 1x1 conv
    """

    def __init__(self, in_channel, out_channel, space_reduction):
        super().__init__()
        self.bottleneck = None

        # if channel does not match or space does not match, add a bottle neck
        if in_channel != out_channel or space_reduction != 1:
            self.bottleneck = conv(in_channel, out_channel, ksize=1, stride=space_reduction)

    def forward(self, x: Tensor, identity: Tensor):

        if self.bottleneck is None:
            return x + identity
        else:
            return x + self.bottleneck(identity)


class BasicBlock(nn.Module):
    """
    Basic resnet layer
    Two conv layers with relu and batch norm and skip connection.
    original mode:      Conv-BN-Relu-Conv-BN-Add-Relu
    preactivation mode: BN-Relu-Conv-BN-Relu-Conv-Add
    """

    def __init__(self, block_name, n_stacks, in_channels, space_reduce, feature_expand, mode='original'):
        super().__init__()

        self.mode = mode
        self.names = ["{}-{}-{}".format(block_name, mode, i+1) for i in range(n_stacks)]
        n_features = in_channels*feature_expand

        for i in range(n_stacks):
            if i > 0:  # only the first layer will apply the reduction in space and/or expansion in features
                in_channels = n_features
                space_reduce = 1
            if mode == 'original':
                conv1 = conv(in_channels, n_features, stride=space_reduce)
                bn1 = nn.BatchNorm2d(n_features)
                relu1 = nn.ReLU(inplace=True)
                conv2 = conv(n_features, n_features)
                bn2 = nn.BatchNorm2d(n_features)
                add = SkipConnection(in_channels, n_features, space_reduce)
                relu2 = nn.ReLU(inplace=True)
                layers = [conv1, bn1, relu1, conv2, bn2, add, relu2]
            elif mode == 'preactivation':
                bn1 = nn.BatchNorm2d(in_channels)
                relu1 = nn.ReLU(inplace=True)
                conv1 = conv(in_channels, n_features, stride=space_reduce)
                bn2 = nn.BatchNorm2d(n_features)
                conv2 = conv(n_features, n_features)
                relu2 = nn.ReLU(inplace=True)
                add = SkipConnection(in_channels, n_features, space_reduce)
                layers = [bn1, relu1, conv1, bn2, conv2, relu2, add]
            else:
                raise NotImplemented("Mode {} is not supported".format(mode))
            self.__setattr__(self.names[i], nn.Sequential(*layers))

    def forward(self, x):

        for stack_name in self.names:
            identity = x
            for layer in self.__getattr__(stack_name):
                layer: nn.Module
                if isinstance(layer, SkipConnection):
                    x = layer(x, identity)
                else:
                    x = layer(x)
        return x


class Flatten(nn.Module):
    """
    Dummy flatten operation
    """
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor):
        x = torch.flatten(x, self.axis)
        return x


class AvgPoolHead(nn.Module):
    """
    Two FC layer head
    global avgpool - flatten - fc - scores
    """

    def __init__(self, block_name, in_channels, n_classes):
        super().__init__()

        self.name = block_name
        avgpool = nn.AdaptiveAvgPool2d((1, 1))    # BxCx1x1
        flatten = Flatten(axis=1)                 # BxC
        fc = nn.Linear(in_channels, n_classes)
        self.__setattr__(block_name, nn.Sequential(avgpool, flatten, fc))

    def forward(self, x):

        for layer in self.__getattr__(self.name):
            x = layer(x)
        return x


class ResNet(nn.Module):

    def __init__(self, settings):
        super().__init__()
        self.layers = []

        for key in settings:
            config = settings[key]
            block = config[0]
            block_name = config[1]
            params = config[1:]
            self.layers.append(block_name)
            self.__setattr__(block_name, block(*params))

    def forward(self, x):

        for block_name in self.layers:
            x = self.__getattr__(block_name)(x)
        return x

