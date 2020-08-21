import torch
from tqdm import tqdm_notebook
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from metrics.metric import Metric
from logs.writer import Writer


class Trainer(object):

    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 callbacks=None,
                 device=None):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = callbacks
        self.device = device

    def _iter(self, batch, gradient_clip, is_train=True):
        raise NotImplementedError('You must implement this method')

    def _update_callbacks(
            self,
            loss,
            y_true,
            y_pred,
            global_step,
            is_train=True,
            **kwargs):

        result = {}
        field_name = 'Test'
        if is_train:
            field_name = 'Train'

        result[field_name + '_loss'] = loss

        for callback in self.callbacks:
            if isinstance(callback, Metric):
                score = callback.step(y_true, y_pred)
                result[field_name + '_' + callback.__class__.__name__] = score

        for callback in self.callbacks:
            if isinstance(callback, Writer):
                callback.step(result, field_name, global_step)
            elif isinstance(callback, Metric):
                pass
            else:
                callback.step()

    def _log_training(self, epoch):
        for callback in self.callbacks:
            if isinstance(callback, Writer):
                callback.execute(epoch)

    def _reset_metrics(self, start_epoch = True):
        if start_epoch:
            for callback in self.callbacks:
                if isinstance(callback, Metric) or isinstance(callback, Writer):
                    callback.reset()
        else:
            for callback in self.callbacks:
                if isinstance(callback, Metric):
                    callback.reset()

    def train(
            self,
            epochs,
            train_loader,
            test_loader=None,
            gradient_clip=50):

        train_global_step = 0
        test_global_step = 0

        for i in range(epochs):

            train_loss = []
            test_loss = []

            self._reset_metrics(start_epoch= True)
            self.model.train()

            for batch in tqdm(train_loader):

                train_global_step += 1
                loss, y_true, y_pred = self._iter(
                    batch, gradient_clip, is_train=True)
                train_loss.append(loss)

                if self.callbacks is not None:
                    self._update_callbacks(
                        loss, y_true, y_pred, train_global_step, is_train=True)

            if test_loader is not None:

                self._reset_metrics(start_epoch=False)
                self.model.eval()

                for batch in tqdm(test_loader):

                    test_global_step += 1

                    loss, y_true, y_pred = self._iter(
                        batch, gradient_clip, is_train=False)
                    test_loss.append(loss)

                    if self.callbacks is not None:
                        self._update_callbacks(
                            loss, y_true, y_pred, test_global_step, is_train=False)

            train_average_loss = sum(train_loss) / len(train_loss)
            if test_loader is not None:
                test_average_loss = sum(test_loss) / len(test_loss)

            self._log_training(i + 1)

