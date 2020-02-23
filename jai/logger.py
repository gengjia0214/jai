from torch.nn import Module
from torch import Tensor
from .dataset import Evaluator
import matplotlib.pyplot as plt
import numpy as np
import torch
import csv
import os


class BasicLogger:

    def __init__(self, log_dst: str, prefix: str, evaluator: Evaluator, resume: bool, keep='one_best',
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

        if not resume:
            for log_dir in [self.model_dst, self.confusion_dst, self.fail_log_dst, loss_log_dst]:
                if not os.path.isdir(log_dir):
                    os.mkdir(log_dir)
                else:
                    raise IOError("Directory {} already exist. Make sure the log_dst is a clean directory!".format(
                        log_dir))

        blt_p = os.path.join(loss_log_dst, "{}_batch_train_log.csv".format(prefix))
        ble_p = os.path.join(loss_log_dst, "{}_batch_eval_log.csv".format(prefix))
        self.bl_p = {"train": blt_p, "eval": ble_p}
        self.ep_p = os.path.join(loss_log_dst, "{}_epoch_log.csv".format(prefix))

        self.fail_log = {}
        self.confusion_matrices = {'train': {}, 'eval': {}}
        self.best_acc = {'avg': 0}
        self.running_loss = {'train': [0, 0], 'eval': [0, 0]}
        self.evaluator = evaluator
        for phase in ['train', 'eval']:
            for name, n_classes in self.evaluator.items():
                self.confusion_matrices[phase][name] = np.zeros((n_classes, n_classes), dtype=np.int)

        if not resume:
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
            temp = self._compute_scores()
            train_loss = round(self.running_loss['train'][0] / self.running_loss['train'][1], 8)
            eval_loss = round(self.running_loss['eval'][0] / self.running_loss['eval'][1], 8)
            self._update_epoch_logs(train_loss, eval_loss, temp)

            # if find better acc, return the logs
            if temp['eval']['avg'] > self.best_acc['avg'] + 1e-8:
                self.best_acc = temp['eval']
                self.export_fail_log()
                self.export_confusion_matrix()
                better = True

            # setup attributes for new epoch
            # if done with all epochs stop
            if phase == 'stop':
                self._report_epoch(train_loss, eval_loss, temp)
                self.curr_epoch = epoch
                return better
            else:
                self._report_epoch(train_loss, eval_loss, temp)
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

    @staticmethod
    def get_failures(fp):
        """
        Get the failures of an epoch
        :param fp: file path
        :return: list of failed classification (ID)
        """

        failures = []
        with open(fp, mode='r') as csv_file:
            reader = csv.reader(csv_file)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                failures.append({'id': row[0], 'name': row[1], 'prediction': row[2], 'truth': row[3]})
        return failures

    def export_best_model(self, model: Module, optimizer, scheduler=None):

        if self.keep == 'one_best':
            name = 'best'
        else:
            name = self.curr_epoch - 1

        mp = os.path.join(self.model_dst, "{}_model_epoch={}.pth".format(self.prefix, name))
        torch.save(model.state_dict(), mp)
        op = os.path.join(self.model_dst, "{}_optimizer_epoch={}.pth".format(self.prefix, name))
        torch.save(optimizer.state_dict(), op)
        if scheduler:
            sp = os.path.join(self.model_dst, "{}_scheduler_epoch={}.pth".format(self.prefix, name))
            torch.save(scheduler.state_dict(), sp)

        if self.verbose:
            print("Model, Scheduler and Optimizer at Epoch {} Was Exported.".format(self.curr_epoch))

    def export_fail_log(self):

        if self.keep == 'one_best':
            name = 'best'
        else:
            name = self.curr_epoch
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
            print("Failure Log at Epoch {} Was Exported.".format(self.curr_epoch))

    def export_confusion_matrix(self):

        if self.keep == 'one_best':
            suffix = 'best'
        else:
            suffix = self.curr_epoch

        for phase in ['train', 'eval']:
            for pname, cfm in self.confusion_matrices[phase].items():
                fn = "{}_confusion_phase={}_predictor={}_epoch={}.npy".format(self.prefix, phase, pname, suffix)
                cfm_p = os.path.join(self.confusion_dst, fn)
                np.save(cfm_p, cfm)
                if self.verbose:
                    print("Confusion Matrices at Epoch {} Was Exported.".format(self.curr_epoch))

    def _create_logs(self):

        with open(self.bl_p['train'], mode='w') as blt_csv:
            head = ["Batch", "Loss"]
            writer = csv.writer(blt_csv)
            writer.writerow(head)

        with open(self.bl_p['eval'], mode='w') as ble_csv:
            head = ["Batch", "Loss"]
            writer = csv.writer(ble_csv)
            writer.writerow(head)

        mapping = {'precision': 'pcs', 'accuracy': 'acc', 'recall': 'recall'}
        ctr = mapping[self.evaluator.criteria]
        with open(self.ep_p, mode='w') as epoch_csv:
            score_head = ["{}_{}_{}".format(phase, name, ctr) for phase in ['train', 'eval'] for name in
                          self.evaluator.names]
            head = ["Epoch", "train_loss", "eval_Loss"] + score_head
            head.append("train_avg_score")
            head.append("valid_avg_score")
            writer = csv.writer(epoch_csv)
            writer.writerow(head)

    def _update_batch_logs(self, phase, batch, loss):

        with open(self.bl_p[phase], mode='a') as bl_csv:
            writer = csv.writer(bl_csv)
            writer.writerow([str(batch), str(round(loss, 8))])

    def _update_epoch_logs(self, train_loss, eval_loss, acc_dict):

        with open(self.ep_p, mode='a') as epoch_csv:
            writer = csv.writer(epoch_csv)
            scores = ["{}".format(acc_dict[phase][name]) for phase in ['train', 'eval'] for name in
                      self.evaluator.names]
            scores.append("{}".format(acc_dict['train']['avg']))
            scores.append("{}".format(acc_dict['eval']['avg']))
            row = [str(self.curr_epoch), str(train_loss), str(eval_loss)] + scores
            writer.writerow(row)

    def _log_predictions(self, phase, outputs: Tensor, truths: Tensor, entry_ids):

        # if there is only one predictor
        if truths.ndim == 1:
            truths = [truths]  # just wrap with a list, this is like N_pred x B ~ 1xB
        else:  # if there are multiple predictor, need to transpose the truth to N_pred x B
            truths = truths.transpose(1, 0)
        for i, name in enumerate(self.evaluator.names):

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

    def _compute_scores(self):
        """
        Compute the scores defined by the Evaluator
        :return: accuracy dictionary
        """

        temp = {'train': {}, 'eval': {}}
        for phase in ['train', 'eval']:
            pooled = []
            for name, cm in self.confusion_matrices[phase].items():
                cm: np.ndarray
                score = self.evaluator.compute_score(cm)
                temp[phase][name] = score
                pooled.append(score)
            avg = self.evaluator.avg_over_predictor(pooled)
            temp[phase]['avg'] = avg
        return temp

    def _report_epoch(self, train_loss, eval_loss, acc_dict):
        """
        Working on new epoch
        :param train_loss: training loss
        :param eval_loss: evaluation loss
        :param acc_dict: accuracy dictionary
        :return: void
        """

        acc_report = {'train': "", 'eval': ""}
        for phase in ['train', 'eval']:
            for name, acc in acc_dict[phase].items():
                if name != 'avg':
                    acc_report[phase] += "Predictor {} Score: {}; ".format(name, acc)
            acc_report[phase] += " AVG Acc: {:<10}".format(acc_dict[phase]['avg'])

        print("\n===================================\n")
        print("Epoch {} Completed".format(self.curr_epoch))
        print("TrainLoss: {} ValidLoss: {}".format(train_loss, eval_loss))
        print("Train: {}".format(acc_report['train']))
        print("Valid: {}".format(acc_report['eval']))

    def _prepare_next_epoch(self):
        """
        Set up for the new epoch
        :return: void
        """

        self.fail_log = {}
        self.running_loss = {'train': [0, 0], 'eval': [0, 0]}
        for phase in ['train', 'eval']:
            for name, n_classes in self.evaluator.items():
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
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Epoch Loss")
        plt.show()


class AdvanceLogger(BasicLogger):
    """
    Basic Logger will log additional information such as activations, etc.
    """

    pass

