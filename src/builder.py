"""
ConvNet Assembler

Author: Jia Geng
Email: gjia0214@gmail.com | jxg570@miami.edu
"""

import os
import pickle
from collections import OrderedDict
from elements.blocks import *
from elements.head import *


class _BaseConfig:
    """
    Base configuration class
    """

    def __init__(self, *args):
        self.model_config = None

    def save_config(self, dst_p: str):
        """
        Save the configuration
        :param dst_p: dst path
        :return: void
        """

        assert os.path.isdir(os.path.dirname(dst_p)), 'Invalid directory'
        assert dst_p.endswith('pkl'), 'dst_p must end with .pkl'

        with open(dst_p, 'wb') as file:
            pickle.dump(obj=self.get_config(), file=file)

        print('Model configuration exported.')

    def load_config(self, src_p):
        """
        Load the model configuration
        :param src_p: src path
        :return: void
        """

        with open(src_p, 'rb') as file:
            self.model_config = pickle.load(file=file)

        print('Model configuration loaded.')

    def get_config(self):
        """
        Get the configuration
        :return: model configuration
        """

        assert isinstance(self.model_config, list), 'Configuration was not set up yet.'
        return self.model_config


class ConfigResNet(_BaseConfig):
    """
    Default ResNet configuration.
    The head module is not configurable. Use AvgPool with 1x1 target space + single FC decoder
    """

    def __init__(self, n_classes: int,
                 init_num_feature_maps: int, init_conv_stride: int, init_kernel_size: int, init_maxpool_stride: int, init_maxpool_size: int,
                 num_blocks: int, num_res_unit_per_block: int, bb_feature_expansion=2, bb_space_reduction=2, num_in_channels=3):
        """
        Constructor.
        :param n_classes: number of classes to be predicted
        :param init_num_feature_maps: number of feature maps for the initial block output (commonly 16, 32, or 64)
        :param init_conv_stride: initial block conv layer stride (1 or 2)
        :param init_maxpool_stride: initial block maxpool stride
        :param init_kernel_size: kernel size for the initial block conv layer (usually 3x3, 5x5, 7x7 depends on the input size)
        :param init_maxpool_size: kernel size for the initial block maxpool layer (2x2, 3x3, 4x4, 5x5)
        :param num_blocks: number of blocks (3 or 4)
        :param num_res_unit_per_block: number of residual unit (one unit is 2x conv-relu-bn layers) per block (commonly 2 or 3)
        :param bb_feature_expansion: block to block feature expansion (default is 2)
        :param bb_space_reduction: block to block space reduction (default is 2)
        :param num_in_channels: number of input channels, default is 3 for RGB, for gray-scale need to set to 1
        """

        super().__init__()

        # prepare the args for the initial block
        init_block_args = {'num_out_channels': init_num_feature_maps, 'conv_stride': init_conv_stride, 'conv_ksize': init_kernel_size,
                           'pool_stride': init_maxpool_stride, 'pool_ksize': init_maxpool_size, 'num_in_channels': num_in_channels}
        self.model_config = [('init_block', init_block_args)]

        # keep track of num_feature_maps
        num_feature_maps = init_num_feature_maps

        # residual block
        for i in range(num_blocks):
            # prepare the residual block params
            res_block_args = {'num_units': num_res_unit_per_block, 'num_in_channels': num_feature_maps, 'init_conv_stride': bb_space_reduction,
                              'feature_expand_ratio': bb_feature_expansion, 'ksize': 3}
            self.model_config.append(('residual_block_{}'.format(i+1), res_block_args))
            num_feature_maps = num_feature_maps * bb_feature_expansion

        # head module
        # n_classes: int, in_channels: int, buffer_reduction: int or None, avgpool_target_shape = (1, 1)
        head_block_args = {'n_classes': n_classes, 'in_channels': num_feature_maps, 'buffer_reduction': None}
        self.model_config.append(('avgfc_head_block', head_block_args))


class ConfigDenseNet(_BaseConfig):

    def __init__(self):
        super().__init__()
        pass


class ConfigMobileNet(_BaseConfig):

    def __init__(self):
        super().__init__()
        pass


class Builder:
    """
    ConvNet builder.
    Take the config object and build the model.
    """

    def __init__(self):
        """
        Constructor
        """

        self.memo = {'init_block': InitConvBlock,
                     'residual_block': ResidualBlock,
                     'avgfc_head_block': AvgPoolFCHead}

    def assemble(self, config: _BaseConfig):

        blocks = OrderedDict()
        config = config.get_config()
        # for each block
        for block_name, block_args in config:
            block_name_parsed = block_name[:block_name.find('block')+5]
            # construct the block module
            module = self.memo[block_name_parsed](**block_args)
            blocks[block_name] = module

        return nn.Sequential(blocks)
