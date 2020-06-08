from typing import List, Union

import torch
import torchbearer
import math
from math import sqrt

import numpy as np
from scipy.sparse import coo_matrix
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import Sampler, Dataset
from torch.utils.data.dataloader import DataLoader
from torchbearer.callbacks import Callback
from torchbearer.callbacks.torch_scheduler import TorchScheduler, StepLR
from torch.optim.optimizer import Optimizer, required
from torch.nn.init import _calculate_fan_in_and_fan_out


class SparseTensorLoss(nn.Module):
    def __init__(self, loss: nn.Module) -> None:
        super().__init__()
        self._wrapped_loss = loss

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if target.layout == torch.sparse_coo:
            target = target.to_dense()
        return self._wrapped_loss.forward(input, target)


class FasterBatchSampler(Sampler):
    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False,
    ):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    @property
    def num_samples(self):
        if not hasattr(self, "_num_samples"):
            self._num_samples = len(self.data_source)
        return self._num_samples

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            iter_list: List[int] = torch.randperm(self.num_samples).tolist()
        else:
            iter_list: List[int] = list(range(self.num_samples))
        for i in range(0, self.num_samples, self.batch_size):
            last_idx = i + self.batch_size
            if last_idx < self.num_samples or not self.drop_last:
                yield iter_list[i:last_idx]


class NoAutoCollationDataLoader(DataLoader):
    @property
    def _auto_collation(self):
        return False

    @property
    def _index_sampler(self):
        return self.batch_sampler


def load_torch_model_training_from_task_dir(model_cls, task_dir: str):
    model_training = model_cls(**get_params(task_dir))
    model_training._output_path = task_dir
    return model_training


def load_torch_model_training_from_task_id(model_cls, task_id: str):
    task_dir = get_task_dir(model_cls, task_id)

    return load_torch_model_training_from_task_dir(model_cls, task_dir)


def lecun_normal_init(tensor: torch.Tensor):
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    nn.init.normal_(tensor, 0, sqrt(1.0 / fan_in))


def he_init(tensor: torch.Tensor):
    nn.init.kaiming_normal_(tensor, mode="fan_in")


def deep_ndarray_to_tensor(batch, device, dtype):
    """ Method to call :func:`to` on tensors or tuples. All items in tuple will have :func:_deep_to called

    :param batch: The mini-batch which requires a :func:`to` call
    :type batch: tuple, list, np.ndarray
    :param device: The desired device of the batch
    :type device: torch.device
    :param dtype: The desired datatype of the batch
    :type dtype: torch.dtype
    :return: The moved or casted batch
    :rtype: tuple, list, torch.Tensor
    """
    is_tuple = isinstance(batch, tuple)

    if isinstance(batch, list) or isinstance(batch, tuple):
        batch = list(batch)
        for i in range(len(batch)):
            batch[i] = deep_ndarray_to_tensor(batch[i], device, dtype)
        batch = tuple(batch) if is_tuple else batch
    else:
        batch = torch.from_numpy(batch).to(device, dtype)

    return batch
