"""
Architecture Manipulator.
All Methods with Exp suffix is not tested
"""

import torch.nn as nn


def replace(model: nn.Module, old, new, depth=-1):
    """
    Search through a neural nets and replace one type of Module with another type of Module
    :param model: neural nets
    :param old: original Module type
    :param new: new Module type
    :param depth: how many layers will be replaced. -1 for all
    :return: void
    """

    if depth == 0:
        return 0

    for name, module in model.named_children():
        if isinstance(module, old):
            if depth != -1:
                depth -= 1
            setattr(model, name, new)
            print("Changed the module from {} to {}".format(name, new))
        else:
            depth = replace(module, old, new, depth)

    return depth


class ReducedPowerResNetExp:
    """
    Experimental
    """

    def __init__(self, r, n_channels, base_model):
        pass

#