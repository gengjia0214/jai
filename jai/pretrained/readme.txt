model-cifar10-resnet18-9377

config:

resized to 49x49
transforms.RandomCrop(49, padding=4)
transforms.RandomHorizontalFlip()
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

one cycle A: 30 + 40 follow up
torch.optim.SGD
lr=0.01,
momentum=0.9
weight_decay=5e-4

one cycle B: 75 + 75 follow up
torch.optim.SGD
lr=0.01,
momentum=0.9
weight_decay=5e-4


