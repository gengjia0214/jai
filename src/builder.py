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


def get_init_block_arg_dict(init_num_feature_maps, init_conv_stride, init_kernel_size, init_maxpool_stride, init_maxpool_size, num_in_channels):
    """
    Wrap the args for init block into a dict.
    Check Config classes for arg doc string
    :return: initial block args dictionary
    """

    # prepare the args for the initial block
    init_block_args = {'num_out_channels': init_num_feature_maps, 'conv_stride': init_conv_stride, 'conv_ksize': init_kernel_size,
                       'pool_stride': init_maxpool_stride, 'pool_ksize': init_maxpool_size, 'num_in_channels': num_in_channels}

    return init_block_args


def get_head_block_arg_dict(n_classes, num_feature_maps, buffer_reduction, avgpool_target_size):
    """
    Wrap the args for the head block into a dict.
    Check Config classes for arg doc string
    :return: head block args dictionary
    """

    head_block_args = {'n_classes': n_classes, 'in_channels': num_feature_maps,
                       'buffer_reduction': buffer_reduction, 'avgpool_target_shape': avgpool_target_size}
    return head_block_args


def sanity_check(arg, num_blocks, var_name):
    """
    Sanity check on the per block arg. When the per block arg was passed as list, need to make sure the length of the list is equal to the number of
    blocks.
    :param arg: per block arg
    :param num_blocks: number of block
    :param var_name: variable name
    :return: void
    """

    if isinstance(arg, list): assert len(arg) == num_blocks, 'Param {} length does not match'.format(var_name)


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
    ResNet configuration.
    """

    def __init__(self, n_classes: int,
                 init_num_feature_maps: int, init_conv_stride: int, init_kernel_size: int, init_maxpool_stride: int, init_maxpool_size: int,
                 num_res_blocks: int, num_res_unit_per_block: int or list, bb_feature_expansion: int or list, bb_space_reduction: int or list,
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
        :param num_res_blocks: Number of blocks (3 or 4)
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
        sanity_check(num_blocks=num_res_blocks, arg=bb_feature_expansion, var_name='bb_feature_expansion')
        sanity_check(num_blocks=num_res_blocks, arg=bb_space_reduction, var_name='bb_space_reduction')
        sanity_check(num_blocks=num_res_blocks, arg=num_res_unit_per_block, var_name='num_res_unit_per_block')

        # prepare the args for the initial block
        init_block = get_init_block_arg_dict(init_num_feature_maps=init_num_feature_maps, init_conv_stride=init_conv_stride,
                                             init_kernel_size=init_kernel_size, init_maxpool_stride=init_maxpool_stride,
                                             init_maxpool_size=init_maxpool_size, num_in_channels=num_in_channels)
        self.model_config = [('init_block', init_block)]

        # keep track of num_feature_maps
        num_feature_maps = init_num_feature_maps

        # residual block
        for i in range(num_res_blocks):

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
        head_block_args = get_head_block_arg_dict(n_classes=n_classes, num_feature_maps=num_feature_maps, buffer_reduction=None,
                                                  avgpool_target_size=avgpool_target_size)
        self.model_config.append(('avgfc_head_block', head_block_args))


class ConfigDenseNet(_BaseConfig):
    """
    Densenet configuration
    """

    def __init__(self, n_classes: int,
                 init_num_feature_maps: int, init_conv_stride: int, init_kernel_size: int, init_maxpool_stride: int, init_maxpool_size: int,
                 num_dense_blocks: int, num_dense_unit_per_block: int or list, growth_rates: int or list, base_bottleneck_sizes: int or list,
                 dropout_rate: float, avgpool_target_size: int, num_in_channels=3, kernel_sizes=3):
        """
        Constructor
        :param n_classes: Number of classes
        :param init_num_feature_maps: Number of feature maps for the initial block output (commonly 16, 32, or 64)
        :param init_conv_stride: Initial block conv layer stride (1 or 2)
        :param init_maxpool_stride: Initial block maxpool stride
        :param init_kernel_size: Kernel size for the initial block conv layer (usually 3x3, 5x5, 7x7 depends on the input size)
        :param init_maxpool_size: Kernel size for the initial block maxpool layer (2x2, 3x3, 4x4, 5x5)
        :param num_dense_blocks: Number of dense block, usually is set to 4. After i-th block, num_features = init_num_feature_maps + growth_rate * i
        :param num_dense_unit_per_block: Number of dense unit per block. Pass int for fixed arg, pass list for per block configuration.
        :param growth_rates: Growth rate for each block. Growth rate decide the increase of feature maps. Common choice are (16, 32, 48)Pass int for
        fixed arg, pass list for per block configuration.
        :param base_bottleneck_sizes: Multiplicative factor for number of bottle neck layers (i.e. base_bottleneck_size * k features in the bottleneck
        layer). Common choice is 4.
        :param dropout_rate: Dropout rate, dropout layer is at the end of each dense block
        :param avgpool_target_size: Average pooling layer output size.
        :param num_in_channels: Number of input channels, default is 3 for RGB, for gray-scale need to set to 1
        :param kernel_sizes: kernel size, default is 3
        """

        super().__init__()

        # sanity check
        sanity_check(num_blocks=num_dense_blocks, arg=num_dense_unit_per_block, var_name='num_dense_unit_per_block')
        sanity_check(num_blocks=num_dense_blocks, arg=growth_rates, var_name='growth_rates')
        sanity_check(num_blocks=num_dense_blocks, arg=base_bottleneck_sizes, var_name='base_bottleneck_sizes')
        sanity_check(num_blocks=num_dense_blocks, arg=kernel_sizes, var_name='kernel_sizes')

        # prepare the args for the initial block
        init_block = get_init_block_arg_dict(init_num_feature_maps=init_num_feature_maps, init_conv_stride=init_conv_stride,
                                             init_kernel_size=init_kernel_size, init_maxpool_stride=init_maxpool_stride,
                                             init_maxpool_size=init_maxpool_size, num_in_channels=num_in_channels)
        self.model_config = [('init_block', init_block)]

        # keep track of num_feature_maps
        num_feature_maps = init_num_feature_maps

        # residual block
        for i in range(num_dense_blocks):
            # check the input type
            growth_rate = growth_rates[i] if isinstance(growth_rates, list) else growth_rates
            base_bottleneck_size = base_bottleneck_sizes[i] if isinstance(base_bottleneck_sizes, list) else base_bottleneck_sizes
            num_unit = num_dense_unit_per_block[i] if isinstance(num_dense_unit_per_block, list) else num_dense_unit_per_block
            kernel_size = kernel_sizes[i] if isinstance(kernel_sizes, list) else kernel_sizes

            # num_layers: int, num_in_channels: int, growth_rate: int,
            # base_bottleneck_size: int, ksize: int, drop_rate: float

            # prepare the residual block params
            dense_block_args = {'num_units': num_unit, 'num_in_channels': num_feature_maps,
                                'base_bottleneck_size': base_bottleneck_size, 'growth_rate': growth_rate,
                                'ksize': kernel_size, 'dropout_rate': dropout_rate}

            # collect the args
            self.model_config.append(('dense_block_{}'.format(i + 1), dense_block_args))

            # update feature map count
            num_feature_maps += growth_rate

        # head module
        head_block_args = get_head_block_arg_dict(n_classes=n_classes, num_feature_maps=num_feature_maps,
                                                  buffer_reduction=None, avgpool_target_size=avgpool_target_size)
        self.model_config.append(('avgfc_head_block', head_block_args))


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
                     'dense_block': DenseBlock,
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
