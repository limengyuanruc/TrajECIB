import logging
import os
import time
from typing import Optional, List, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.utils.rnn as rnn_utils

from utils.logger import get_logger

class BaseTrainer():
    def __init__(
            self,
            model: nn.Module,
            dataloader_dict,
            base_lr: float,
            steps,
            lr_decay_ratio,
            log_dir: str,
            n_exp: int,
            save_iter: int = 300,
            clip_grad_value: Optional[float] = None,
            max_epochs: Optional[int] = 1000,
            patience: Optional[int] = 1000,
            device: Optional[Union[torch.device, str]] = None,
    ):
        super().__init__()

        self._logger = get_logger(
            log_dir, __name__, 'info_{}.log'.format(n_exp), level=logging.INFO)
        if device is None:
            print("`device` is missing, try to train and evaluate the model on default device.")
            if torch.cuda.is_available():
                print("cuda device is available, place the model on the device.")
                self._device = torch.device("cuda")
            else:
                print("cuda device is not available, place the model on cpu.")
                self._device = torch.device("cpu")
        else:
            if isinstance(device, torch.device):
                self._device = device
            else:
                self._device = torch.device(device)
        self._model = model.to(self._device)

        self._loss_fn = nn.CrossEntropyLoss()
        self._loss_fn.to(self._device)
        self._base_lr = base_lr
        self._optimizer = Adam(self.model.parameters(), base_lr)

        if lr_decay_ratio == 1:
            self._lr_scheduler = None
        else:
            self._lr_scheduler = MultiStepLR(self.optimizer,
                                             steps,
                                             gamma=lr_decay_ratio)
        self._clip_grad_value = clip_grad_value
        self._max_epochs = max_epochs
        self._patience = patience
        self._save_iter = save_iter
        self._save_path = log_dir
        self._n_exp = n_exp
        self._dataloader_dict = dataloader_dict


    @property
    def model(self):
        return self._model


    @property
    def dataloader_dict(self):
        return self._dataloader_dict
    
    @property
    def data(self):
        return self._dataloader_dict

    @property
    def logger(self):
        return self._logger

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def loss_fn(self):
        return self._loss_fn

    @property
    def device(self):
        return self._device

    @property
    def save_path(self):
        return self._save_path


    def _to_numpy(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.cpu().detach().numpy() for tensor in tensors]
        else:
            return tensors.cpu().detach().numpy()

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [Tensor(array) for array in nparray]
        else:
            return Tensor(nparray)

    def save_model(self, epoch, save_path, n_exp):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # filename = 'epoch_{}.pt'.format(epoch)
        filename = 'final_model_{}.pt'.format(n_exp)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
        return filename
        # return True

    def load_model(self, epoch, save_path, n_exp):
        filename = 'final_model_{}.pt'.format(n_exp)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename),map_location=self.model.augs.device))
        return True

    def early_stop(self, epoch, best_acc):
        self.logger.info('Early stop at epoch {}, val accuracy = {:.6f}'.format(epoch, best_acc))

    def _check_device(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)