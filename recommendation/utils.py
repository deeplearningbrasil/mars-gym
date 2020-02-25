import os
import ast
from datetime import datetime
from multiprocessing.pool import Pool
from typing import List, Union, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from math import sqrt
from torch.nn.init import _calculate_fan_in_and_fan_out
from tqdm import tqdm

from recommendation.files import get_params, get_task_dir

"""
Url: https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8
"""
import unicodedata
import string

valid_filename_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
char_limit = 255
def clean_filename(filename, whitelist=valid_filename_chars, replace=' '):
    # replace spaces
    for r in replace:
        filename = filename.replace(r,'_')
    
    # keep only valid ascii chars
    cleaned_filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode()
    
    # keep only whitelisted chars
    cleaned_filename = ''.join(c for c in cleaned_filename if c in whitelist)
    if len(cleaned_filename)>char_limit:
        print("Warning, filename truncated because it was over {}. Filenames may no longer be unique".format(char_limit))
    return cleaned_filename[:char_limit]    

# test
s='fake_folder/\[]}{}|~`"\':;,/? abcABC 0123 !@#$%^&*()_+ clᐖﯫⅺຶ 讀ϯՋ㉘ ⅮRㇻᎠ⍩ 𝁱C ℿ؛ἂeuឃC ᅕ ᑉﺜͧ bⓝ s⡽Հᛕ\ue063 牢𐐥er ᐛŴ n წş .ھڱ                                 df                                         df                                  dsfsdfgsg!zip'
clean_filename(s) # 'fake_folder_abcABC_0123_____clxi_28_DR_C_euC___bn_s_er_W_n_s_.zip'

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


def literal_eval_if_str(element):
    if isinstance(element, str):
        return ast.literal_eval(element)
    return element


def _parallel_literal_eval(series: Union[pd.Series, np.ndarray], pool: Pool, use_tqdm: bool = True) -> list:
    if use_tqdm:
        return list(tqdm(pool.map(literal_eval_if_str, series), total=len(series)))
    else:
        return pool.map(literal_eval_if_str, series)


def date_to_day_of_week(date: str) -> int:
    return int(datetime.strptime(date, '%Y-%m-%d').strftime('%w'))


def date_to_day_of_month(date: str) -> int:
    return int(datetime.strptime(date, '%Y-%m-%d').strftime('%d'))


def datetime_to_shift(datetime_: str) -> str:
    datetime_: datetime = datetime_ if isinstance(datetime_, datetime) else datetime.strptime(datetime_,
                                                                                              '%Y-%m-%d %H:%M:%S')
    day_of_week = int(datetime_.strftime('%w'))
    # 0 - 4:59h - dawn
    # 5 - 9:59h - breakfast
    # 10 - 13:59h - lunch
    # 14 - 16:59h - snack
    # 17 - 23:59h - dinner
    if 0 <= datetime_.hour <= 4:
        shift = "dawn"
    elif 5 <= datetime_.hour <= 9:
        shift = "breakfast"
    elif 10 <= datetime_.hour <= 13:
        shift = "lunch"
    elif 14 <= datetime_.hour <= 16:
        shift = "snack"
    else:
        shift = "dinner"
    return "%s %s" % ("weekday" if day_of_week < 4 else "weekend", shift)


def get_scores_per_tuples(account_idx: int, merchant_idx_list: List[int],
                          scores_per_tuple: Dict[Tuple[int, int], float]) -> List[float]:
    return list(map(lambda merchant_idx: scores_per_tuple.get((account_idx, merchant_idx), -1.0), merchant_idx_list))


def get_scores_per_tuples_with_click_timestamp(account_idx: int, merchant_idx_list: List[int],
                                               click_timestamp: datetime,
                                               scores_per_tuple: Dict[Tuple[int, int, datetime], float]) -> List[float]:
    return list(map(lambda merchant_idx: scores_per_tuple.get((account_idx, merchant_idx, click_timestamp), -1.0),
                    merchant_idx_list))
