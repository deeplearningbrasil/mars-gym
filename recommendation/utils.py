import ast
import os
from math import sqrt
from multiprocessing.pool import Pool
from typing import List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out
from tqdm import tqdm

from recommendation.files import get_params, get_task_dir

def load_torch_model_training_from_task_dir(model_cls,
                                            task_dir: str):
    model_training = model_cls(**get_params(task_dir))
    model_training._output_path = task_dir
    return model_training


def load_torch_model_training_from_task_id(model_cls,
                                           task_id: str):
    task_dir = get_task_dir(model_cls, task_id)

    return load_torch_model_training_from_task_dir(model_cls, task_dir)


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


def chunks(l: Union[list, range], n: int) -> Union[list, range]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def parallel_literal_eval(series: Union[pd.Series, np.ndarray], pool: Pool = None, use_tqdm: bool = True) -> list:
    if pool:
        return _parallel_literal_eval(series, pool, use_tqdm)
    else:
        with Pool(os.cpu_count()) as p:
            return _parallel_literal_eval(series, p, use_tqdm)


def _parallel_literal_eval(series: Union[pd.Series, np.ndarray], pool: Pool, use_tqdm: bool = True) -> list:
    if use_tqdm:
        return list(tqdm(pool.map(ast.literal_eval, series), total=len(series)))
    else:
        return pool.map(ast.literal_eval, series)
