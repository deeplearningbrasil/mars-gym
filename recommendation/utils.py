from math import sqrt
from typing import List

import torch
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out


def lecun_normal_init(tensor: torch.Tensor):
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    nn.init.normal_(tensor, 0, sqrt(1. / fan_in))


def he_init(tensor: torch.Tensor):
    nn.init.kaiming_normal_(tensor, mode='fan_in')


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