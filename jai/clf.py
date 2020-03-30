import numpy as np
import torch
import cv2 as cv
import PIL.Image as im
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch.nn as nn


class JaiClassifier:

    def __init__(self, arch: nn.Module, model_state_path, device='cuda', encoding=None):
        """
        constructor
        :param arch: architecture
        :param model_state_path: model state path
        :param device: mounting device
        :param encoding: encoding dictionary if provided
        """

        self.encoding = encoding
        self.arch = arch
        model_state = torch.load(model_state_path, map_location='cpu')
        self.arch.load_state_dict(model_state)
        for param in self.arch.parameters():
            param.requires_grad = False
        self.arch.eval()
        self.arch.to(device)

    def predict(self, X, order='BCHW', prob=False, mode='single'):
        """
        make prediction
        :param X:
        :param y: ground truth if provided
        :param order:
        :param mode: model for prediction, single or multi crop/flip
        :param prob: whether to return probability
        :return:
        """

        proba = None
        if order == 'BCHW':
            pass
        if order == 'BHWC':
            X = torch.from_numpy(X)
            X = X.permute(0, 3, 1, 2)
        if order == 'HWC':
            X = torch.from_numpy(X)
            X = X.permute(2, 0, 1)
            X = X.unsqueeze(0)
        print(X.shape)
        output = self.arch(X)

        if prob:
            proba = nn.functional.softmax(output, dim=1)

        labels = torch.argmax(output, dim=1)
        labels = labels.detach().to('cpu').tolist()
        proba = proba.detach().to('cpu')

        if self.encoding is not None:
            if not prob:
                labels = [(label, self.encoding[label]) for label in labels]
            else:
                labels = [(label, self.encoding[label], proba[i][label].tolist()) for i, label in enumerate(labels)]
        else:
            if prob:
                labels = [(label, proba[i][label].tolist()) for i, label in enumerate(labels)]

        return labels, proba


# from torch.utils.data import Dataset, DataLoader
# import torchvision
# from jai.arch.resnet import *
#
# model_path = '/home/jgeng/Documents/Git/jai/test/resnet18-9377-later150/model/cifar10-resnet18-9377.pth'
# config = get_settings(mode='preserve', n_classes=10)
# resnet18 = ResNet(config)
# clf = JaiClassifier(resnet18, model_state_path=model_path)
# root = '/home/jgeng/Documents/Git/jai/jai/cifar10'
# test_transform = transforms.Compose([transforms.Resize((49, 49)),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# test_set = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=test_transform)
# test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
# for batch in test_loader:
#     X, y = batch
#     X = X.to('cuda')
#     labels, proba = clf.predict(X=X, prob=True)
#     break
#
# print(labels, y)