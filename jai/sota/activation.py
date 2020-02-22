"""
SOTA Activations

Currently Supported Sota Activations
- Mish https://arxiv.org/abs/1908.08681
"""


import torch
from torch.nn import Module
import torch.nn.functional as F


class Mish(Module):
    """
    A modified implementation of
    Mish: A Self Regularized Non-Monotonic Neural Activation Function by Misra
    ref: https://arxiv.org/abs/1908.08681
    org repo: https://github.com/digantamisra98/Mish.git

    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """

    def __init__(self, inplace=False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, x):

        x = x * torch.tanh(F.softplus(x))
        return x

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


