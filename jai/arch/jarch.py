import torch.nn as nn


class JaiArch(nn.Module):
    """
    Abstract class
    """

    def __init__(self):
        super().__init__()

    def freeze(self, *args):
        raise NotImplemented("Need to implement freeze layer method!")

    def unfreeze(self, *args):
        raise NotImplemented("Need to implement unfreeze layer method!")

