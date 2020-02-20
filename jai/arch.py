"""
Architecture Manipulator.
"""

from fastai.basics import *
import torchvision.models as models
from jai.borrowed import Mish


class Outlet(nn.Module):
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
    # TODO: might need to fine tune the architectures

    def __init__(self, arch=models.densenet121, n_classes=(168, 11, 7), pre=True, dropout_p=0.5):
        super().__init__()
        m = arch(True) if pre else arch()

        # TODO: first layer is 7x7x1 -> 64 features
        # TODO: might be better to use a smaller kernel with lass features 5x5x1 -> 8    features
        conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # TODO: whether this is ok need to be further explored, should use the Gray Scale Transform of the weight
        # TODO: RGB[A] to Gray:Y←0.299⋅R+0.587⋅G+0.114⋅B
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
    def __init__(self):
        super().__init__()

    def forward(self, inp: tuple, targets, reduction='mean'):
        # TODO: multiple independent output with unbalanced n_classes
        # TODO: how should we sum the losses from each class? - does not really matter for the mulgate

        x1, x2, x3 = inp
        x1, x2, x3 = x1.float(), x2.float(), x3.float()
        y = targets.long()

        loss1 = 2 * F.cross_entropy(x1, y[:, 0], reduction=reduction)
        loss2 = 1 * F.cross_entropy(x2, y[:, 1], reduction=reduction)
        loss3 = 1 * F.cross_entropy(x3, y[:, 2], reduction=reduction)
        combined = loss1 + loss2 + loss3
        return combined
        # combined = 0
        # ratio = [2, 1, 1]
        # for r, x, y in zip(ratio, inp, targets):
        #     loss = r * F.cross_entropy(x.float(), y, reduction=reduction)
        #     combined += loss
        #
        # return combined

