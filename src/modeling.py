"""
Modeling Pipeline

Author: Jia Geng
Email: gjia0214@gmail.com | jxg570@miami.edu
"""

import gc
import os
import random

import copy
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim.lr_scheduler import *
from torch.optim.optimizer import Optimizer
from datautils.datahandler import DataHandler
from builder import *
from datetime import datetime
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm as tqdm_terminal
from functools import partial
import numpy as np
import pickle


def compute_metric(ground_truth: list, prediction: list,):
    """
    Compute a bunch of performance metrics based on the key:
    accuracy, precision, recall and f1 score; by class, micro, macro
    :param ground_truth: list of ground truth labels
    :param prediction: list of predicted labels
    :return: performance metrics by class, micro, macro
    """

    # compute performance metrics
    acc = metrics.accuracy_score(y_true=ground_truth, y_pred=prediction)
    precision_by_class = metrics.precision_score(y_true=ground_truth, y_pred=prediction, average=None, zero_division=0)
    precision_micro = metrics.precision_score(y_true=ground_truth, y_pred=prediction, average='micro', zero_division=0)
    precision_macro = metrics.precision_score(y_true=ground_truth, y_pred=prediction, average='macro', zero_division=0)
    recall_by_class = metrics.recall_score(y_true=ground_truth, y_pred=prediction, average=None, zero_division=0)
    recall_micro = metrics.recall_score(y_true=ground_truth, y_pred=prediction, average='micro', zero_division=0)
    recall_macro = metrics.recall_score(y_true=ground_truth, y_pred=prediction, average='macro', zero_division=0)
    f1_by_class = metrics.f1_score(y_true=ground_truth, y_pred=prediction, average=None, zero_division=0)
    f1_micro = metrics.f1_score(y_true=ground_truth, y_pred=prediction, average='micro', zero_division=0)
    f1_macro = metrics.f1_score(y_true=ground_truth, y_pred=prediction, average='macro', zero_division=0)

    metric_dict = {'accuracy': acc,
                   'precision': {'by_class': precision_by_class, 'micro': precision_micro, 'macro': precision_macro},
                   'recall': {'by_class': recall_by_class, 'micro': recall_micro, 'macro': recall_macro},
                   'f1': {'by_class': f1_by_class, 'micro': f1_micro, 'macro': f1_macro}}

    return metric_dict


class _Logger:
    """
    Logger for collecting the loss, predictions and maintain a confusion matrix during model training/evaluation
    """

    def __init__(self, n_classes: int, criteria: str, verbose: bool):
        """
        Constructor
        :param n_classes: number of classes
        :param criteria: criteria for model selection: accuracy, precision, recall and f1 score
        :param verbose: whether to print out the logging process, turn on for debugging
        """

        # At least need to use top-1 acc, should also support one more criteria
        assert criteria in ['accuracy', 'recall', 'precision', 'f1']
        self.criteria = criteria
        self.n_classes = n_classes
        self.verbose = verbose

        # cache best performance
        self.best_criteria_metric = -1
        self.best_acc = -1
        self.best_epoch = -1

        # batch & epoch log
        self.batch_loss = {'train': [], 'eval': []}
        self.epoch_loss = {'train': [], 'eval': []}
        self.epoch_perf = {'train': [], 'eval': []}

        # best epoch log
        # unlock and refresh when meet better epoch
        self.best_comprehensive_metric = None
        self.best_ground_truths = {'train': [], 'eval': []}
        self.best_predictions = {'train': [], 'eval': []}

        # row index for truth, column index for prediction
        # unlock and refresh when meet better epoch
        self.best_confusion_matrix = {'train': np.zeros((n_classes, n_classes)),
                                      'eval': np.zeros((n_classes, n_classes))}

        # temp pointer only contains at data for the current epoch
        self.temp_batch_loss = {'train': [], 'eval': []}
        self.temp_ground_truth = {'train': [], 'eval': []}
        self.temp_prediction = {'train': [], 'eval': []}
        self.temp_confusion_matrix = {'train': np.zeros((n_classes, n_classes)),
                                      'eval': np.zeros((n_classes, n_classes))}

    def login_batch(self, phase: str, ground_truth: list, predictions: list, loss: float or None):
        """
        Login the batch data
        :param phase: train or eval
        :param ground_truth: ground truth
        :param predictions: predicted class
        :param loss: batch loss
        :return: void
        """

        # performance metrics
        for t, p in zip(ground_truth, predictions):
            # update the list container
            self.temp_ground_truth[phase].append(t)
            self.temp_prediction[phase].append(p)
            # row index for truth, column index for prediction
            self.temp_confusion_matrix[phase][t][p] += 1

        # loss log for the current epoch
        # login to the temp as well as the global pointer
        # for test model on a testing set and report performance only, loss input should be none
        if loss is not None:
            self.temp_batch_loss[phase].append(loss)
            self.batch_loss[phase].append(loss)

    def login_iteration(self, phase: str, criteria: str, epoch: int):
        """
        Operations when an iteration complete
        For training phase:
            1. update the epoch log
            2. calculate the performance
            3. reset the temp pointers
        For eval phase:
            1. update the epoch log
            2. calculate the performance
            3. update best perf and best epoch when better model find, send the signal to agent
            4. reset the temp pointers
            5. update the epoch idx
        :param phase: train or eval
        :param criteria: criteria metric key: accuracy, precision, recall or f1 score
        :param epoch: the current epoch index
        :return: epoch_loss, acc, selected_metric
        """

        # flag
        find_better_model = False

        # update the epoch log
        epoch_loss = np.mean(self.temp_batch_loss[phase])
        self.epoch_loss[phase].append(epoch_loss)

        # compute the performance metrics
        perf_metrics = compute_metric(ground_truth=self.temp_ground_truth[phase], prediction=self.temp_prediction[phase])
        acc = perf_metrics['accuracy']
        selected_metric = perf_metrics[criteria]['macro'] if criteria != 'accuracy' else acc

        # for eval phase => reset temp pointer & check whether get better performance
        if phase == 'eval':
            # update best epoch info
            if selected_metric > self.best_criteria_metric:
                self.best_criteria_metric = selected_metric
                self.best_acc = acc
                self.best_epoch = epoch
                self.best_ground_truths = self.temp_ground_truth
                self.best_predictions = self.temp_prediction
                self.best_confusion_matrix = self.temp_confusion_matrix
                self.best_comprehensive_metric = perf_metrics  # login all metrics into this pointer
                find_better_model = True
            # reset the temp pointers
            self.temp_batch_loss = {'train': [], 'eval': []}
            self.temp_ground_truth = {'train': [], 'eval': []}
            self.temp_prediction = {'train': [], 'eval': []}
            self.temp_confusion_matrix = {'train': np.zeros((self.n_classes, self.n_classes)),
                                          'eval': np.zeros((self.n_classes, self.n_classes))}
            # collect the garbage
            gc.collect()

        return epoch_loss, acc, selected_metric, find_better_model

    def load(self, src: dict or str):
        """
        Load the logger from dictionary or a pickle file
        This is called by the modeling agents
        :param src: logger state dict or a pickle file path
        :return: void
        """

        if isinstance(src, str):          # load the pickle file
            if not src.endswith('pth'):
                raise Exception('src path should end with .pth')
            with open(src, 'rb') as file:
                state_dict = pickle.load(file=file)
        elif isinstance(src, dict):
            state_dict = src
        else:
            raise Exception('src need to be either the logger state dict or a path to the pickle file')

        for attr, val in state_dict.items():
            self.__setattr__(attr, val)

    def save(self, dst_pth: str,):
        """
        Save the logger state. This method usually won't used by users.
        The modeling agents will handle the serialization from surface
        :param dst_pth: dst folder
        :return: void
        """

        if not dst_pth.endswith('pth'):
            raise Exception('dst_pth should end with .pth')

        with open(dst_pth, 'wb') as file:
            pickle.dump(obj=self.get_logger_state(), file=file)

    def get_logger_state(self):
        """
        Get the logger state
        :return: logger state dict
        """

        return self.__dict__

    def get_best_perf(self):
        """
        Getter to get the best perf metric, i.e. auc
        :return: selected performance metric
        """

        return self.best_criteria_metric

    def get_best_confusion_matrix(self):
        """
        Getter to get the confusion matrix of the best epoch
        :return: confusion matrix
        """

        return self.best_confusion_matrix

    def get_best_predictions(self):
        """
        Getter to get the predictions of the best epoch
        :return: ground truth and predictions
        """

        return self.best_ground_truths, self.best_predictions

    def reset(self):
        """
        Re-initialize the logger
        :return: void
        """

        self.__init__(n_classes=self.n_classes, criteria=self.criteria, verbose=self.verbose)


class __BaseAgent:
    """
    Agent interface class
    Parent class for Trainer and Evaluator
    """

    def __init__(self, model: nn.Module or ModelConfig, loss_module: nn.Module or None,
                 n_classes: int, criteria: str, verbose: bool,
                 optimizer: partial or None, scheduler: partial or None,
                 prefix: str, checkpoint_folder: str,
                 new_head=None, blocks_to_freeze=None,
                 *args):
        """
        Abstract class of Base Agent.
        Check child class doc for param details.
        """

        if not os.path.isdir(checkpoint_folder):
            raise FileExistsError('File does not exits: {}'.format(checkpoint_folder))

        # agent type
        self.agent = None
        self.running_env = 'notebook'
        self.pbar = {'notebook': tqdm_notebook, 'terminal': tqdm_terminal}

        # key model settings
        # set up the model config instance if provided
        if isinstance(model, nn.Module):
            self.model = model
        elif isinstance(model, ModelConfig):
            self.model = Builder.assemble(model)
        else:
            raise Exception('Model must be either an nn.Module or a Config instance')
        self.loss_module = loss_module
        self.n_classes = n_classes
        self.criteria = criteria
        self.device = None

        # optimizer & scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

        # check point setting
        self.base_epoch = 0
        self.checkpoint_folder = checkpoint_folder
        self.prefix = prefix

        # set up args for transfer learning
        # when new_head_args is not None, transfer learning is activated
        # the model config instance will be modified during initialization
        self.new_head = new_head
        self.blocks_to_freeze = blocks_to_freeze if blocks_to_freeze is not None else []
        if self.new_head is not None:
            # sanity check
            error_msg1 = "new_head_args must be a nn.module or a dictionary, use builder.get_head_block_args() to get the arg dict"
            assert isinstance(self.new_head, dict) or isinstance(self.new_head, nn.Module), error_msg1
            error_msg2 = "When new head is provided, the model arg must be a ModelConfig instead of nn.Module."
            assert not isinstance(self.model, ModelConfig), error_msg2

        # initialize logger
        self.logger = _Logger(n_classes=n_classes, criteria=criteria, verbose=verbose)

    def initialize(self, device: str, pretrained_param_path=None, n_threads=12):
        """
        Initialize the model, optimizer and the scheduler.
        Put the model on the specified device.
        Optionally, load the pretrained params and replace the head module (if new_head is not None) for transfer learning.
        :param device: which device to be trained on e.g. 'cpu' - on cpu, 'cuda:0' - on gpu 0, 'cuda:1' - on gpu 1
        :param pretrained_param_path: path to pretrained model param
        :param n_threads: configure the thread usage, only applicable when using cpu for computing
        :return: void
        """

        if device == 'cpu':
            torch.set_num_threads(n_threads)

        # move model to device before initialize the optimizers and scheduler
        self.model = self.model.to(device)
        self.loss_module = self.loss_module.to(device)
        self.device = device

        # initialize the optimizer and scheduler
        if self.optimizer is not None:
            self.optimizer = self.optimizer(self.model.parameters())

        # use scheduler
        if self.scheduler is not None:
            self.scheduler = self.scheduler(optimizer=self.optimizer)

        # load the pretrained param if provided
        if pretrained_param_path is not None:
            # sanity check
            error_msg = 'pretrained_param_path must end with .pth'
            assert isinstance(pretrained_param_path, str) and pretrained_param_path.endswith('pth'), error_msg
            meta_state = torch.load(f=pretrained_param_path, map_location='cpu')

            # load the model state
            # compatible with pure model state or the logger output
            model_state = meta_state['model_state'] if 'model_state' in meta_state else meta_state
            self.model.load_state_dict(model_state)

            # replace the old head with new head
            if self.new_head is not None:

                # sanity check, the model must have the head_block
                error_msg = 'Model does not have the head_block. Head replacement only supports the models created by the Builder class'
                assert 'head_block' in [name[0] for name in self.model.named_modules()], error_msg

                # set it to the new head
                new_head = AvgPoolFCHead(**self.new_head) if isinstance(self.new_head, dict) else self.new_head
                new_head = new_head.to(self.device)
                self.model.head_block = new_head
                print('Head module has been replaced.')

            # turn on/off the requires_grads
            for (param_name, param) in self.model.named_parameters():
                requires_grad = True
                for frozen_block in self.blocks_to_freeze:
                    if param_name.startswith(frozen_block):  # decide which block to freeze based on the prefix
                        requires_grad = False
                        print("Param {} requires_grad was turned off".format(param_name))
                param.requires_grad = requires_grad

    def load_model_params(self, model_state_dict: dict):
        """
        Load pre-trained model params
        :param model_state_dict: model state dict
        :return: void
        """

        self.model.load_state_dict(model_state_dict)

    def load_checkpoint(self):
        """
        Load the model check point directly from the checkpoint folder
        :return: void
        """

        # load metastate
        metastate_dict = torch.load(os.path.join(self.checkpoint_folder, "{}_metastate_best.pth".format(self.prefix)),
                                    map_location=torch.device('cpu'))

        # load model param
        self.model.load_state_dict(metastate_dict['model_state'])

        # only load the optimizer, scheduler and logger for trainer
        if self.agent == 'trainer':
            # load the epoch number
            self.base_epoch = metastate_dict['epoch']

            # load optimizer and scheduler
            if self.optimizer is not None:
                self.optimizer: Optimizer
                self.optimizer.load_state_dict(metastate_dict['optimizer_state'])

            if self.scheduler is not None:
                self.scheduler.load_state_dict(metastate_dict['scheduler_state'])

            # load the logger
            self.logger.load(metastate_dict['logger_state'])

    def save_checkpoint(self, *args):
        pass

    def get_logger(self):
        """
        Getter to get the Logger
        :return: Logger
        """

        return self.logger

    def reset_logger(self):
        """
        Reset the logger
        :return: void
        """

        self.logger.reset()

    def use_notebook(self):
        """
        Switch to notebook mode - affect the tqdm pbar
        :return: void
        """

        self.running_env = 'notebook'

    def use_terminal(self):
        """
        Switch to terminal mode - affect the tqdm pbar
        :return: void
        """

        self.running_env = 'terminal'


class Trainer(__BaseAgent):
    """
    TODO: tutorials
    TODO: make it able to choose micro/macro for model selection
    """

    def __init__(self, model: nn.Module or Builder or ModelConfig, loss_module: nn.Module,
                 n_classes: int, criteria: str,
                 optimizer: partial or torch.optim.optimizer.Optimizer, scheduler: partial or None,
                 prefix: str, checkpoint_folder: str,
                 new_head=None, blocks_to_freeze=None, verbose=False):
        """
        Constructor
        :param model: model architecture (nn.Module) or a Builder/Config Instance that builds the model. For transfer learning, pass the Config
        Instance.
        :param loss_module: loss module
        :param n_classes: number of classes
        :param criteria: criteria for model selection: accuracy, precision, recall and f1 score (will use macro average)
        :param optimizer: parameter optimizer. DO NOT pass the instance (i.e. DO NOT do encoder_optimizer=Adam(...)) instead, just pass the Class
        interface (i.e. encoder_optimizer=Adam) or, if need to specify the optimizer params, use partial: (e.g. encoder_optimizer=partial(Adam,
        lr=1e-5, ...)) instance initialization will be handled by the trainer
        :param scheduler: learning rate scheduler. pass the interface instead of the instance
        :param prefix: checkpoint naming prefix
        :param checkpoint_folder: checkpoint
        :param new_head: arg for transfer learning. If pass an nn.Module, the module will replace the current head block. The
        :param blocks_to_freeze: arg for transfer learning. Pass the block prefix to freeze the blocks, e.g. ['init_block'] will freeze all
        parameters whose name starting with 'init_block'.
        :param verbose: whether to print out additional message for debugging
        """

        super().__init__(model=model, loss_module=loss_module,
                         n_classes=n_classes, criteria=criteria, verbose=verbose,
                         optimizer=optimizer, scheduler=scheduler,
                         prefix=prefix, checkpoint_folder=checkpoint_folder,
                         new_head=new_head, blocks_to_freeze=blocks_to_freeze)
        self.agent = 'trainer'

    def train(self, datahandler: DataHandler, epochs, seed, allowed_loss_ratio: float or None):
        """
        Model training pipeline with loss & performance metrics logging & automatic checkpoint
        :param datahandler: data handler
        :param epochs: total number of training epochs
        :param seed: seed for random state
        :param allowed_loss_ratio: set up a train/eval loss ratio threshold, when the loss ratio becomes larger than the threshold, stop the training
        :return:
        """

        assert isinstance(self.optimizer, Optimizer), 'Need to call initialize() before train()!'

        # fix random state
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # fix the cudnn backend (might break the code if pytorch refactored this)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # memory monitoring
        memory_usage = {'train': -1, 'eval': -1}

        # placeholder
        epoch_recorder = {'train': {},
                          'eval': {}}

        # main loop
        for epoch in self.pbar[self.running_env](range(self.base_epoch, epochs), total=epochs - self.base_epoch, desc='Epochs'):
            print("=====Start Epoch {}======\n".format(epoch))

            # loss cache
            loss_cache = {'train': 0, 'eval': 0}

            for phase in ['train', 'eval']:

                # switch phase setting - only collect the feature/temporal importance during trainig time
                if phase == 'train':
                    self.model.train()
                    self.loss_module.train()
                else:
                    self.model.eval()
                    self.loss_module.eval()  # might not be necessary as the loss function does not hold any params

                pbar_msg = 'Epoch {} Phase {}'.format(epoch, phase)
                for i, mini_batch in self.pbar[self.running_env](enumerate(datahandler[phase]), total=len(datahandler[phase]), desc=pbar_msg):

                    # zero the gradient
                    if phase == 'train':
                        self.optimizer.zero_grad()

                    # grab X, Y
                    if isinstance(mini_batch, dict):
                        X_mini_batch, Y_mini_batch = mini_batch['x'], mini_batch['y']
                    else:
                        X_mini_batch, Y_mini_batch = mini_batch

                    # move to computing device
                    X_mini_batch = X_mini_batch.to(self.device)
                    Y_mini_batch = Y_mini_batch.to(self.device)

                    # enable the gradient flow if in training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        # run through model and get the output_scores
                        output_scores = self.model(X_mini_batch)
                        # TODO: refactor the loss_module forward pass
                        loss = self.loss_module(X=output_scores, Y=Y_mini_batch)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # batch logging - ground truth, prediction, batch loss
                    loss_cpu = loss.cpu().detach().tolist()
                    ground_truth = Y_mini_batch.cpu().detach().view(-1).tolist()
                    predictions = output_scores.cpu().detach().argmax(-1).view(-1).tolist()
                    self.logger.login_batch(phase=phase, ground_truth=ground_truth, predictions=predictions,
                                            loss=loss_cpu)
                    loss_cache[phase] += loss_cpu

                if self.device != 'cpu':
                    memory_usage[phase] = torch.cuda.memory_allocated(device=self.device)

                # clear the memory
                del loss
                del output_scores
                del X_mini_batch
                del Y_mini_batch

                # ITERATION END - login the compute the performance metrics
                epoch_loss, acc, selected_metric, find_better_model = self.logger.login_iteration(phase=phase, criteria=self.criteria, epoch=epoch)
                epoch_recorder[phase]['loss'] = np.round(epoch_loss, 4)
                epoch_recorder[phase]['acc'] = np.round(acc, 4)
                epoch_recorder[phase]['perf'] = np.round(selected_metric, 4)

            # EPOCH END - done with train & eval
            # report
            print("+++ Epoch {} Report +++".format(epoch))
            print("Train Loss: {} Train Accuracy: {} Train {}: {}".format(epoch_recorder['train']['loss'], epoch_recorder['train']['acc'],
                                                                          self.criteria, epoch_recorder['train']['perf']))
            print("Eval Loss: {} Eval Accuracy: {} Eval {}: {}".format(epoch_recorder['eval']['loss'], epoch_recorder['eval']['acc'],
                                                                       self.criteria, epoch_recorder['eval']['perf']))
            if self.device != 'cpu':
                print("Memory usage during train: {} | Memory usage during eval: {}".format(memory_usage['train'], memory_usage['eval']))

            # scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()

            # checkpoint
            if find_better_model:
                print('+++')
                print("Better model found at epoch={}, eval accuracy={}, eval {}={}".format(epoch, acc, self.criteria, selected_metric))
                self.save_checkpoint(epoch=epoch, last_epoch=False)
                print('Checkpoint saved.')
                print('+++')
            print("\n======End Epoch {}=======\n".format(epoch))

            loss_ratio = loss_cache['eval'] / loss_cache['train']

            # for last epoch
            if epoch == epochs - 1 or (allowed_loss_ratio is not None and loss_ratio > allowed_loss_ratio):
                self.save_checkpoint(epoch, last_epoch=True)
            if allowed_loss_ratio is not None and loss_ratio > allowed_loss_ratio:
                print('Loss ratio beyond the threshold, training stopped.')
                break

    def save_checkpoint(self, epoch, last_epoch=False):
        """
        Save the model check point
        :param epoch: epoch number
        :param last_epoch: saving mode for last epoch
        :return: void
        """

        # epoch number, logger, model, optimizer and scheduler
        metastate_dict = {'epoch': epoch + 1,
                          'logger_state': self.logger.get_logger_state(),
                          'model_state': self.model.state_dict()}

        # optimizer and scheduler
        if self.optimizer is not None:
            self.optimizer: Optimizer
            metastate_dict['optimizer_state'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            metastate_dict['scheduler_state'] = self.scheduler.state_dict()

        # model structure text and training date
        string_rep = str(self.model)
        with open(os.path.join(self.checkpoint_folder, '{}_model_arch.txt'.format(self.prefix)), 'w') as fp:
            now = datetime.now()
            now = now.strftime("%m/%d/%Y, %H:%M:%S")
            fp.writelines(now + '\n')
            fp.writelines(string_rep)

        # save the meta state
        if not last_epoch:  # an additional checkpoint for the last epoch
            torch.save(metastate_dict, os.path.join(self.checkpoint_folder, "{}_metastate_best.pth".format(self.prefix)))
        else:  # the best epoch so far
            torch.save(metastate_dict, os.path.join(self.checkpoint_folder, "{}_metastate_last.pth".format(self.prefix)))


class Evaluator(__BaseAgent):
    """
    TODO: describe workflow
    """

    def __init__(self, model: nn.Module,
                 n_classes: int, criteria: str,
                 prefix: str, checkpoint_folder: str,
                 verbose=False):
        """
        Constructor
        :param model: model architecture
        :param n_classes: number of classes
        :param criteria: criteria for model selection: accuracy, precision, recall and f1 score (will use macro average)
        :param prefix: checkpoint naming prefix
        :param checkpoint_folder: checkpoint directory
        :param verbose: whether to print out additional message for debugging
        """

        super().__init__(model=model, loss_module=None,
                         n_classes=n_classes, criteria=criteria, verbose=verbose,
                         optimizer=None, scheduler=None,
                         prefix=prefix, checkpoint_folder=checkpoint_folder)
        self.agent = 'evaluator'

    def evaluate(self, datahandler: DataHandler, manual_load_model_param=False):
        """
        Evaluation on the data.
        The evaluation pipeline will compute auc, acc, roc curve and optionally feature rank.
        :param datahandler: data handler, data should be loaded on the eval key
        :param manual_load_model_param: whether the model param was manually loaded, if not, will load it from the check point folder
        :return: acc, auc, roc curve, auc_by_seq_len, feature_ranker object (if fit_ranker=True)
        """

        # load the model
        if not manual_load_model_param:
            self.load_checkpoint()

        # main loop
        for i, mini_batch in self.pbar[self.running_env](enumerate(datahandler['eval']), total=len(datahandler['eval']), desc='Evaluation'):

            # grab X, Y
            X_mini_batch, Y_mini_batch = mini_batch['x'], mini_batch['y']

            # ground truth
            ground_truth = Y_mini_batch.cpu().view(-1).tolist()

            # move to computing device
            X_mini_batch = X_mini_batch.to(self.device)

            # run through model and get the output_scores
            output_scores = self.model(X_mini_batch)
            prediction = output_scores.detach().argmax(-1).cpu().view(-1).tolist()

            # batch logging - ground truth, prediction, batch loss
            self.logger.login_batch(phase='eval', ground_truth=ground_truth, predictions=prediction,
                                    loss=None)

        # ITERATION END - login the compute the performance metrics
        self.logger.login_iteration(phase='eval', criteria=self.criteria, epoch=0)
        self.report_evaluation_results()

        # output
        output = {'performance': self.get_comprehensive_metrics(),
                  'confusion_matrix': self.get_confusion_matrix()}

        return output

    def report_evaluation_results(self):
        """
        Report the evaluation results
        :return: void
        """

        # report
        print("Performance Report {}".format(self.prefix))
        print("Accuracy: {}".format(self.logger.best_acc))
        print("{} : {}".format(self.criteria.capitalize(), self.logger.best_criteria_metric))

    def get_accuracy(self):
        """
        Getter to get the evaluation accuracy
        :return: void
        """

        return self.logger.best_acc

    def get_comprehensive_metrics(self):
        """
        Getter to get the evaluation auc
        :return: performance metric dict
        """

        return copy.deepcopy(self.logger.best_comprehensive_metric)

    def get_confusion_matrix(self):
        """
        Getter to get the confusion matrix
        :return: confusion matrix
        """

        return self.logger.best_confusion_matrix['eval'].copy()
