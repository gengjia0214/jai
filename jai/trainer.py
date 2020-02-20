from .logger import *
from .augments import Augment
from tqdm.notebook import tqdm
import torch as torch
from torch.nn import Module
from functools import partial
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler


class BasicTrainer:
    # TODO: refactor augmentation. Gradually change aug prob.
    def __init__(self, model: Module, optimizer: partial, scheduler: partial = None,
                 device=torch.device('cuda'), verbose=False):
        """
        Constructor
        :param model: module
        :param optimizer: partial function of optimizer, do not include the model parameter
        :param scheduler: partial function of scheduler, do not include the optimizer
        """

        assert isinstance(optimizer, partial), "optimizer should be a partial func with all args except the " \
                                               "parameters."
        assert isinstance(scheduler, partial), "scheduler should be a partial func with all args except the optimizer."

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.verbose = verbose

    def load_model_state(self, state_dict: dict):
        """
        Load model state dict
        :param state_dict: model state dict
        :return: void
        """

        self.model: Module
        self.model.load_state_dict(state_dict)

    def initialize(self, optimizer_state: dict = None, scheduler_state: dict = None):
        """
        Initialize the optimizer and scheduler.
        Optionally load the state dict for optimizer and the scheduler
        :param optimizer_state: optimizer state dict
        :param scheduler_state: scheduler state dict
        :return: void
        """

        self.model: Module
        self.model.to(self.device)
        self.optimizer = self.optimizer(self.model.parameters())
        assert isinstance(self.optimizer, Optimizer), "fail to initialize the optimizer, check the input!"

        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
            if self.verbose:
                print("Optimizer state was loaded!")

        if self.scheduler:
            self.scheduler = self.scheduler(self.optimizer)
            assert isinstance(self.scheduler, Scheduler), "fail to initialize the scheduler, check the input!"

        if scheduler_state:
            self.scheduler.load_state_dict(scheduler_state)
            if self.verbose:
                print("Scheduler state was loaded!")

    def train(self, train_dataloader: DataLoader, eval_dataloader: DataLoader, epochs: int, loss_func: Module,
              logger: BasicLogger, augment: Augment = None):
        """
        Train model
        :param train_dataloader: train data
        :param eval_dataloader: eval data
        :param epochs: total number of epochs
        :param loss_func: loss function
        :param logger: logger
        :param augment: augmentation class
        :return: void
        """

        assert isinstance(self.optimizer, Optimizer), "Need to initialize before training!"
        assert isinstance(self.scheduler, Scheduler), "Need to initialize before training!"
        assert isinstance(loss_func, Module), "Loss function should be implemented as torch Module!"
        assert isinstance(logger, BasicLogger), "logger should be a Logger Instance!"

        if not isinstance(logger, BasicLogger):
            raise NotImplemented("Currently only support basic logger.")

        data_loaders = {'train': train_dataloader, 'eval': eval_dataloader}
        pbar_epoch = tqdm(total=epochs, desc='Epoch')
        for epoch in range(epochs):

            for phase in ['train', 'eval']:
                # TODO: try to refactor the augment, currently need to pass it to both the train method and the
                #  dataset class. maybe refactor the split method in dataset
                if augment:
                    augment.switch_phase(phase)
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                # mini_batch: {'x': tensor for input data, 'y': tensor for ground truth}
                for i, mini_batch in enumerate(data_loaders[phase]):
                    # these should already be in tensor format
                    inputs = mini_batch['x'].to(self.device)
                    truths = mini_batch['y'].to(self.device)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # enable the gradient flow if in training mode
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = loss_func(outputs, truths)  # the loss function should work with batch data

                        # log the data
                        # support collecting data id
                        entry_ids = None
                        if 'id' in mini_batch:
                            entry_ids = mini_batch['id']
                        batch = len(data_loaders[phase]) * epoch + i

                        # if the model is better, export it
                        if logger.receive(epoch, batch, phase, loss, outputs, truths, entry_ids):
                            logger.export_best_model(epoch-1, self.model, self.optimizer, self.scheduler)
                            print("Better Model Found At Epoch {}.".format(epoch))
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            self.scheduler.step(None)  # weird warning if remove the None here
            pbar_epoch.update(1)
        # last dummy epoch
        logger.receive(epochs, batch=-1, phase='stop')



