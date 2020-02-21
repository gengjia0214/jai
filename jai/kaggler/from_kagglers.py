"""
A bit of everything developed by the mighty kagglers and tweaked by me.
"""

from fastai.basics import *
import torchvision.models as models


class MishFunction(torch.autograd.Function):
    """
    I forgot where i got this...
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x))
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))


class Mish(nn.Module):
    """
    I forgot where i got this...
    """

    def forward(self, x):
        return MishFunction.apply(x)


def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)


class Outlet(nn.Module):
    """
    Modified based on:
    org. author: lafoss
    org. src: https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-lb-0-964
    """

    def __init__(self, n_in, n_out, dropout_p):

        super().__init__()
        # TODO: might not need dropout for densnet
        # AdaConcat -> Mish -> Flatten
        # -> BN -> Dropout -> FC (n_out = n_classes)
        layers = [AdaptiveConcatPool2d(), Mish(), Flatten(),
                  *bn_drop_lin(n_in * 2, 512, True, dropout_p, Mish()),
                  *bn_drop_lin(512, n_out, True, dropout_p)]

        self.fc = nn.Sequential(*layers)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        return self.fc(x)


class Dnet1Ch(nn.Module):
    """
    Single Channel NN + Multiple output using DenseNet121 backbone.
    Modified based on:
    org. author: lafoss
    org. src: https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-lb-0-964
    """

    # TODO: might need to fine tune the architectures
    def __init__(self, arch=models.densenet121, n_classes=(168, 11, 7), pre=True, dropout_p=0.5):
        super().__init__()
        m = arch(True) if pre else arch()

        # TODO: first layer is 7x7x1 -> 64 features
        # TODO: might be better to use a smaller kernel with lass features 5x5x1 -> 8    features
        conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # TODO: whether this is ok need to be further explored, might need to use the Gray Scale weight
        # TODO: RGB[A] to Gray: Y←0.299⋅R+0.587⋅G+0.114⋅B
        w = (m.features.conv0.weight.sum(1)).unsqueeze(1)
        conv0.weight = nn.Parameter(w, requires_grad=True)

        self.layer0 = nn.Sequential(conv0, m.features.norm0, nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            m.features.denseblock1)
        self.layer2 = nn.Sequential(m.features.transition1, m.features.denseblock2)
        self.layer3 = nn.Sequential(m.features.transition2, m.features.denseblock3)
        self.layer4 = nn.Sequential(m.features.transition3, m.features.denseblock4,
                                    m.features.norm5)

        nc = self.layer4[-1].weight.shape[0]
        self.head1 = Outlet(nc, n_classes[0], dropout_p)
        self.head2 = Outlet(nc, n_classes[1], dropout_p)
        self.head3 = Outlet(nc, n_classes[2], dropout_p)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)

        return [x1, x2, x3]


class CombineLoss(nn.Module):
    """
    Modified based on:
    org. author: lafoss
    org. src: https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-lb-0-964
    """

    def __init__(self):
        super().__init__()

    def forward(self, inp: tuple, targets, reduction='mean'):

        x1, x2, x3 = inp
        x1, x2, x3 = x1.float(), x2.float(), x3.float()
        y = targets.long()

        # TODO: how should we sum the losses from each class?
        loss1 = 2 * F.cross_entropy(x1, y[:, 0], reduction=reduction)
        loss2 = 1 * F.cross_entropy(x2, y[:, 1], reduction=reduction)
        loss3 = 1 * F.cross_entropy(x3, y[:, 2], reduction=reduction)
        combined = loss1 + loss2 + loss3
        return combined