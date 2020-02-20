from torch.nn import Module
from torch import Tensor
from .datahandler import DataClassDict
import matplotlib.pyplot as plt
import numpy as np
import torch
import csv
import os


class BasicLogger:

    def __init__(self, log_dst: str, prefix: str, class_dict: DataClassDict, keep='one_best',
                 verbose=False):

        if keep not in ['one_best', 'all_best']:
            raise NotImplemented("Currently only support saving the one best or all best models")

        self.verbose = verbose
        self.keep = keep
        self.curr_epoch = 0
        self.prefix = prefix
        self.model_dst = os.path.join(log_dst, 'model')
        self.confusion_dst = os.path.join(log_dst, 'confusion')
        self.fail_log_dst = os.path.join(log_dst, 'failure_log')
        loss_log_dst = os.path.join(log_dst, 'loss_log')

        for log_dir in [self.model_dst, self.confusion_dst, self.fail_log_dst, loss_log_dst]:
            if not os.path.isdir(log_dir):
                os.mkdir(log_dir)

        blt_p = os.path.join(loss_log_dst, "{}_batch_train_log.csv".format(prefix))
        ble_p = os.path.join(loss_log_dst, "{}_batch_eval_log.csv".format(prefix))
        self.bl_p = {"train": blt_p, "eval": ble_p}
        self.ep_p = os.path.join(loss_log_dst, "{}_epoch_log.csv".format(prefix))

        self.fail_log = {}
        self.confusion_matrices = {'train': {}, 'eval': {}}
        self.best_acc = {'avg': 0}
        self.running_loss = {'train': [0, 0], 'eval': [0, 0]}
        self.class_dict = class_dict
        for phase in ['train', 'eval']:
            for name, n_classes in self.class_dict.items():
                self.confusion_matrices[phase][name] = np.zeros((n_classes, n_classes), dtype=np.int)

        self._create_logs()

    def receive(self, epoch, batch, phase, loss: torch.Tensor = None, outputs=None, truths=None, entry_ids=None):
        """
        Receive logs from the trainer.
        :param epoch: epoch number
        :param batch: batch number
        :param phase: train/val phase
        :param loss: loss
        :param outputs: output from the model (before the softmax!)
        :param truths: truth
        :param entry_ids: entry id if applicable
        :return: void
        """

        better = False
        if epoch != self.curr_epoch and (phase in ['train', 'stop']):
            # last epoch completed check whether have better result
            temp = self._compute_acc()
            train_loss = round(self.running_loss['train'][0] / self.running_loss['train'][1], 8)
            eval_loss = round(self.running_loss['eval'][0] / self.running_loss['eval'][1], 8)
            self._update_epoch_logs(epoch - 1, train_loss, eval_loss, temp)

            # if find better acc, return the logs
            if temp['eval']['avg'] > self.best_acc['avg'] + 1e-8:
                self.best_acc = temp['eval']
                self.export_fail_log(epoch-1)
                self.export_confusion_matrix(epoch-1)
                better = True

            # setup attributes for new epoch
            # if done with all epochs stop
            if phase == 'stop':
                self._report_epoch(epoch, train_loss, eval_loss, temp)
                return better
            else:
                self._report_epoch(epoch-1, train_loss, eval_loss, temp)
                self._prepare_next_epoch()
            self.curr_epoch = epoch

        # log the batch loss
        loss = loss.detach().cpu().tolist()
        self._update_batch_logs(phase, batch, loss)
        self.running_loss[phase][0] += loss
        self.running_loss[phase][1] += 1

        # log the predictions to the confusion matrix (if at eval phase)
        self._log_predictions(phase, outputs, truths, entry_ids)

        return better

    def plot(self, item, size=(12, 8), dpi=400):

        if item == 'loss':
            self._plot_loss(size=size, dpi=dpi)
        else:
            raise NotImplemented("Currently only support plotting the losses.")

    def get_failures(self, epoch='best'):
        """
        Get the failures of an epoch
        :param epoch: epoch
        :return: list of failed classification (ID)
        """
        fn = "{}_failures_epoch={}.csv".format(self.prefix, epoch)
        fp = os.path.join(self.fail_log_dst, fn)
        if not os.path.isfile(fp):
            raise FileExistsError("File {} does not exist. Check whether the epoch is correct.".format(epoch))
        failures = []
        with open(fp, mode='r') as csv_file:
            reader = csv.reader(csv_file)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                failures.append({'id': row[0], 'name': row[1], 'prediction': row[2], 'truth': row[3]})
        return failures

    def export_best_model(self, epoch, model: Module, optimizer, scheduler=None):

        if self.keep == 'one_best':
            name = 'best'
        else:
            name = epoch

        mp = os.path.join(self.model_dst, "{}_model_epoch={}.pth".format(self.prefix, name))
        torch.save(model.state_dict(), mp)
        op = os.path.join(self.model_dst, "{}_optimizer_epoch={}.pth".format(self.prefix, name))
        torch.save(optimizer.state_dict(), op)
        if scheduler:
            sp = os.path.join(self.model_dst, "{}_scheduler_epoch={}.pth".format(self.prefix, name))
            torch.save(scheduler.state_dict(), sp)

        if self.verbose:
            print("Model, Scheduler and Optimizer at Epoch {} Was Exported.".format(epoch))

    def export_fail_log(self, epoch):

        if self.keep == 'one_best':
            name = 'best'
        else:
            name = epoch
        fl_p = os.path.join(self.fail_log_dst, "{}_failures_epoch={}.csv".format(self.prefix, name))
        with open(fl_p, mode='w') as failure_csv:
            writer = csv.writer(failure_csv)
            head = ['ID', 'Predictor', 'Prediction', 'Truth']
            writer.writerow(head)
            for entry_id, log in self.fail_log.items():
                row = [str(entry_id)]
                for predictor_name in log:
                    row.append(predictor_name)
                    row.append(log[predictor_name]['prediction'])
                    row.append(log[predictor_name]['truth'])
                writer.writerow(row)
        if self.verbose:
            print("Failure Log at Epoch {} Was Exported.".format(epoch))

    def export_confusion_matrix(self, epoch):

        if self.keep == 'one_best':
            suffix = 'best'
        else:
            suffix = epoch

        for phase in ['train', 'eval']:
            for pname, cfm in self.confusion_matrices[phase].items():
                fn = "{}_confusion_phase={}_predictor={}_epoch={}.npy".format(self.prefix, phase, pname, suffix)
                cfm_p = os.path.join(self.confusion_dst, fn)
                np.save(cfm_p, cfm)
                if self.verbose:
                    print("Confusion Matrices at Epoch {} Was Exported.".format(epoch))

    def _create_logs(self):

        with open(self.bl_p['train'], mode='w') as blt_csv:
            head = ["Batch", "Loss"]
            writer = csv.writer(blt_csv)
            writer.writerow(head)

        with open(self.bl_p['eval'], mode='w') as ble_csv:
            head = ["Batch", "Loss"]
            writer = csv.writer(ble_csv)
            writer.writerow(head)

        with open(self.ep_p, mode='w') as epoch_csv:
            acc_head = ["{}_{}Acc".format(phase, name) for phase in ['train', 'eval'] for name in self.class_dict.names]
            head = ["Epoch", "Train Loss", "Eval Loss"] + acc_head
            head.append("Train_AvgAcc")
            head.append("Valid_AvgACC")
            writer = csv.writer(epoch_csv)
            writer.writerow(head)

    def _update_batch_logs(self, phase, batch, loss):

        with open(self.bl_p[phase], mode='a') as bl_csv:
            writer = csv.writer(bl_csv)
            writer.writerow([str(batch), str(round(loss, 8))])

    def _update_epoch_logs(self, epoch, train_loss, eval_loss, acc_dict):

        with open(self.ep_p, mode='a') as epoch_csv:
            writer = csv.writer(epoch_csv)
            acc = ["{}".format(acc_dict[phase][name]) for phase in ['train', 'eval'] for name in self.class_dict.names]
            acc.append("{}".format(acc_dict['train']['avg']))
            acc.append("{}".format(acc_dict['eval']['avg']))
            row = [str(epoch), str(train_loss), str(eval_loss)] + acc
            writer.writerow(row)

    def _log_predictions(self, phase, outputs: Tensor, truths: Tensor, entry_ids):

        # if there is only one predictor
        if truths.ndim == 1:
            truths = [truths]  # just wrap with a list, this is like N_pred x B ~ 1xB
        else:  # if there are multiple predictor, need to transpose the truth to N_pred x B
            truths = truths.transpose(1, 0)
        for i, name in enumerate(self.class_dict.names):

            # get batch prediction and truth
            predictions = outputs[i].argmax(-1).detach().cpu().int().tolist()
            batch_truths = truths[i].detach().cpu().int().tolist()
            batch_predictions = predictions

            # TODO: this might have a parallel implementation
            for j, (pred, tru) in enumerate(zip(batch_predictions, batch_truths)):
                self.confusion_matrices[phase][name][pred, tru] += 1
                if phase == 'eval':
                    if entry_ids and pred != tru:
                        if entry_ids[j] not in self.fail_log:
                            self.fail_log[entry_ids[j]] = {}
                        self.fail_log[entry_ids[j]] = {name: {'prediction': pred, 'truth': tru}}

    def _compute_acc(self):
        """
        Compute the average accuracy using confusion matrix
        :return: accuracy dictionary
        """

        temp = {'train': {}, 'eval': {}}
        total = 0
        for phase in ['train', 'eval']:
            for name, confusion in self.confusion_matrices[phase].items():
                confusion: np.ndarray
                acc = round(np.sum(confusion.diagonal()) / confusion.sum(), 8)
                temp[phase][name] = acc
                total += acc
            avg_acc = round(total / len(self.class_dict.names), 8)
            temp[phase]['avg'] = avg_acc
        return temp

    def _report_epoch(self, epoch, train_loss, eval_loss, acc_dict):
        """
        Working on new epoch
        :param epoch: next epoch
        :param train_loss: training loss
        :param eval_loss: evaluation loss
        :param acc_dict: accuracy dictionary
        :return: void
        """

        acc_report = {'train': "", 'eval': ""}
        for phase in ['train', 'eval']:
            for name, acc in acc_dict[phase].items():
                if name != 'avg':
                    acc_report[phase] += "Predictor {} Acc: {}; ".format(name, acc)
            acc_report[phase] += " AVG Acc: {:<10}".format(acc_dict[phase]['avg'])

        print("\n===================================\n")
        print("Epoch {} Completed".format(epoch))
        print("TrainLoss: {} ValidLoss: {}".format(train_loss, eval_loss))
        print("Train: {}".format(acc_report['train']))
        print("Valid: {}".format(acc_report['eval']))

        self.fail_log = {}
        self.running_loss = {'train': [0, 0], 'eval': [0, 0]}
        for name, n_classes in self.class_dict.items():
            self.confusion_matrices['eval'][name] = np.zeros((n_classes, n_classes), dtype=np.int)

    def _prepare_next_epoch(self):
        """
        Set up for the new epoch
        :return: void
        """

        self.fail_log = {}
        self.running_loss = {'train': [0, 0], 'eval': [0, 0]}
        for phase in ['train', 'eval']:
            for name, n_classes in self.class_dict.items():
                self.confusion_matrices[phase][name] = np.zeros((n_classes, n_classes), dtype=np.int)

    def _plot_loss(self, size=(12, 4), dpi=400):
        batch_losses = {'train': [], 'eval': []}
        for name, fp in self.bl_p.items():
            with open(fp, mode='r') as bl_train:
                reader = csv.reader(bl_train)
                i = 0
                for row in reader:
                    if i == 0:
                        i += 1
                        continue
                    i += 1
                    batch_losses[name].append(float(row[1]))
        epoch_loss = {'train': [], 'eval': []}

        with open(self.ep_p, mode='r') as ep_log:
            reader = csv.reader(ep_log)
            i = 0
            for row in reader:
                if i == 0:
                    i += 1
                    continue
                i += 1
                epoch_loss['train'].append(float(row[1]))
                epoch_loss['eval'].append(float(row[2]))

        plt.figure(figsize=size, dpi=dpi)
        plt.subplot(3, 1, 1)
        plt.plot(batch_losses['train'], label='Batch Train Loss')
        plt.xlabel("Batch")
        plt.ylabel("Batch Training Loss")
        plt.subplot(3, 1, 2)
        plt.plot(batch_losses['eval'], label='Batch Validation Loss')
        plt.xlabel("Batch")
        plt.ylabel("Batch Validation Loss")
        plt.subplot(3, 1, 3)
        plt.plot(epoch_loss['train'], label='Epoch Train Loss')
        plt.plot(epoch_loss['eval'], label='Epoch Validation Loss')
        plt.legend(loc='top right')
        plt.xlabel("Epoch")
        plt.ylabel("Epoch Loss")
        plt.show()


class AdvanceLogger(BasicLogger):
    """
    Basic Logger will log additional information such as activations, etc.
    """

    def __init__(self, log_dst, prefix, model_dst=None, save_when='best'):
        super().__init__(log_dst, prefix, model_dst, save_when)

    def receive(self, epoch, batch, phase, loss: torch.Tensor = None, outputs=None, truths=None, entry_ids=None):
        pass

