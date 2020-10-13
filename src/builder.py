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
                 num_blocks: int, num_res_unit_per_block: int or list, bb_feature_expansion: int or list, bb_space_reduction: int or list,
                 avgpool_target_size: int, num_in_channels=3):
        """
        Constructor.
        TODO： add per block configuration, i.e. expansion, n_unit, etc
        :param n_classes: number of classes to be predicted
        :param init_num_feature_maps: Number of feature maps for the initial block output (commonly 16, 32, or 64)
        :param init_conv_stride: Initial block conv layer stride (1 or 2)
        :param init_maxpool_stride: Initial block maxpool stride
        :param init_kernel_size: Kernel size for the initial block conv layer (usually 3x3, 5x5, 7x7 depends on the input size)
        :param init_maxpool_size: Kernel size for the initial block maxpool layer (2x2, 3x3, 4x4, 5x5)
        :param num_blocks: Number of blocks (3 or 4)
        :param num_res_unit_per_block: Number of residual unit (one unit is 2x conv-relu-bn layers) per block (commonly 2 or 3)
        :param bb_feature_expansion: Block to block feature expansion. If pass int, use fixed reduction for each block. To specify the expansion
        ratio for each block, pass a list of int.
        :param bb_space_reduction: Block to block space reduction. If pass int, use fixed reduction for each block. To specify the reduction ratio
        for each block, pass a list of int.
        :param avgpool_target_size: Average pooling layer output size.
        :param num_in_channels: Number of input channels, default is 3 for RGB, for gray-scale need to set to 1
        """

        super().__init__()

        # sanity check
        if isinstance(bb_feature_expansion, list): assert len(bb_feature_expansion) == num_blocks, 'Length of bb_feature_expansion does not much ' \
                                                                                                   'with num_blocks'
        if isinstance(bb_space_reduction, list): assert len(bb_space_reduction) == num_blocks, 'Length of bb_space_reduction does not much with ' \
                                                                                               'num_blocks'
        if isinstance(num_res_unit_per_block, list): assert len(num_res_unit_per_block) == num_blocks, 'Length of num_res_unit_per_block does not ' \
                                                                                                       'much with num_blocks'

        # prepare the args for the initial block
        init_block_args = {'num_out_channels': init_num_feature_maps, 'conv_stride': init_conv_stride, 'conv_ksize': init_kernel_size,
                           'pool_stride': init_maxpool_stride, 'pool_ksize': init_maxpool_size, 'num_in_channels': num_in_channels}
        self.model_config = [('init_block', init_block_args)]

        # keep track of num_feature_maps
        num_feature_maps = init_num_feature_maps

        # residual block
        for i in range(num_blocks):

            # check the input type
            expand_ratio = bb_feature_expansion[i] if isinstance(bb_feature_expansion, list) else bb_feature_expansion
            stride = bb_space_reduction[i] if isinstance(bb_space_reduction, list) else bb_space_reduction
            num_unit = num_res_unit_per_block[i] if isinstance(num_res_unit_per_block, list) else num_res_unit_per_block

            # prepare the residual block params
            res_block_args = {'num_units': num_unit, 'num_in_channels': num_feature_maps,
                              'space_reduction_ratio': stride, 'feature_expand_ratio': expand_ratio, 'ksize': 3}

            # collect the args
            self.model_config.append(('residual_block_{}'.format(i+1), res_block_args))

            # update feature map count
            num_feature_maps = num_feature_maps * expand_ratio

        # head module
        head_block_args = {'n_classes': n_classes, 'in_channels': num_feature_maps,
                           'buffer_reduction': None, 'avgpool_target_shape': avgpool_target_size}
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
        """
        Assemble the ConvNet following the configuration
        :param config: config object
        :return: nn module
        """

        blocks = OrderedDict()
        config = config.get_config()
        # for each block
        for block_name, block_args in config:
            block_name_parsed = block_name[:block_name.find('block')+5]
            # construct the block module
            module = self.memo[block_name_parsed](**block_args)
            blocks[block_name] = module

        return nn.Sequential(blocks)
