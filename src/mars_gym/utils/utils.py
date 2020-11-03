import os
import ast
from datetime import datetime, timedelta
from multiprocessing.pool import Pool
from typing import List, Union, Dict, Tuple
from zipfile import ZipFile
#from google.cloud import storage
import json
import scipy
import numpy as np
import pandas as pd
from math import sqrt
from tqdm import tqdm
import shutil
from random import randrange
from pyspark.sql.functions import udf

from mars_gym.utils.files import get_params, get_task_dir

"""
Url: https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8
"""
import unicodedata
import string

valid_filename_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
char_limit = 255


# UDF Spark similar to array_position
def array_index(x, y):
    if x is None:
        return -1

    idxs = [i for i, e in enumerate(x) if e == y]

    if len(idxs) == 0:
        return -1

    return idxs[0]


array_index_udf = udf(array_index)
# array_index_udf = udf(lambda x,y: [i for i, e in enumerate(x) if e==y ])


def random_date(start, l):
    current = start
    while l > 0:
        curr = current + timedelta(minutes=randrange(60))
        yield curr
        l -= 1


def clean_filename(filename, whitelist=valid_filename_chars, replace=" "):
    # replace spaces
    for r in replace:
        filename = filename.replace(r, "_")

    # keep only valid ascii chars
    cleaned_filename = (
        unicodedata.normalize("NFKD", filename).encode("ASCII", "ignore").decode()
    )

    # keep only whitelisted chars
    cleaned_filename = "".join(c for c in cleaned_filename if c in whitelist)
    if len(cleaned_filename) > char_limit:
        print(
            "Warning, filename truncated because it was over {}. Filenames may no longer be unique".format(
                char_limit
            )
        )
    return cleaned_filename[:char_limit]


# test
s = "fake_folder/\[]}{}|~`\"':;,/? abcABC 0123 !@#$%^&*()_+ clᐖﯫⅺຶ 讀ϯՋ㉘ ⅮRㇻᎠ⍩ 𝁱C ℿ؛ἂeuឃC ᅕ ᑉﺜͧ bⓝ s⡽Հᛕ\ue063 牢𐐥er ᐛŴ n წş .ھڱ                                 df                                         df                                  dsfsdfgsg!zip"
clean_filename(s)  # 'fake_folder_abcABC_0123_____clxi_28_DR_C_euC___bn_s_er_W_n_s_.zip'


def chunks(l: Union[list, range], n: int) -> Union[list, range]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def parallel_literal_eval(
    series: Union[pd.Series, np.ndarray], pool: Pool = None, use_tqdm: bool = True
) -> list:
    if pool:
        return _parallel_literal_eval(series, pool, use_tqdm)
    else:
        with Pool(os.cpu_count()) as p:
            return _parallel_literal_eval(series, p, use_tqdm)


def literal_eval_if_str(element):
    if isinstance(element, str):
        return ast.literal_eval(element)
    return element


def _pad_sequence(seq, pad) -> np.ndarray:
    if seq is None:
        return None
    else:
        return (([0] * pad) + seq)[-pad:]


def to_array(xs):
    return (
        [
            literal_eval_if_str(c)
            if isinstance(literal_eval_if_str(c), int)
            else literal_eval_if_str(c)[0]
            for c in literal_eval_if_str(xs)
        ]
        if xs is not None
        else None
    )


def _parallel_literal_eval(
    series: Union[pd.Series, np.ndarray], pool: Pool, use_tqdm: bool = True
) -> list:
    #from IPython import embed; embed()
    if use_tqdm:
        return list(tqdm(pool.map(literal_eval_if_str, series), total=len(series)))
    else:
        return pool.map(literal_eval_if_str, series)


def date_to_day_of_week(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d").strftime("%w"))


def date_to_day_of_month(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d").strftime("%d"))


def get_scores_per_tuples(
    account_idx: int,
    merchant_idx_list: List[int],
    scores_per_tuple: Dict[Tuple[int, int], float],
) -> List[float]:
    return list(
        map(
            lambda merchant_idx: scores_per_tuple.get(
                (account_idx, merchant_idx), -1.0
            ),
            merchant_idx_list,
        )
    )


def get_scores_per_tuples_with_click_timestamp(
    account_idx: int,
    merchant_idx_list: List[int],
    click_timestamp: datetime,
    scores_per_tuple: Dict[Tuple[int, int, datetime], float],
) -> List[float]:
    return list(
        map(
            lambda merchant_idx: scores_per_tuple.get(
                (account_idx, merchant_idx, click_timestamp), -1.0
            ),
            merchant_idx_list,
        )
    )


def get_all_file_paths(directory):
    # initializing empty file paths list
    file_paths = []

    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    # returning all file paths
    return file_paths


def save_trained_data(source_dir: str, target_dir: str):
    print("save_trained_data from '{}' to '{}'".format(source_dir, target_dir))

    file_paths = get_all_file_paths(source_dir)

    # printing the list of all files to be zipped
    print("Following files will be zipped:")
    for file_name in file_paths:
        print(file_name)

    # writing files to a zipfile
    zip_filename = source_dir.split("/")[-1] + ".zip"
    with ZipFile(source_dir + "/" + zip_filename, "w") as zip:
        # writing each file one by one
        for file in file_paths:
            zip.write(file)

    # if "gs://" in target_dir:
    #     bucket = storage.Client().bucket(target_dir.split("//")[-1])
    #     # blob   = bucket.blob('{}/{}'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), zip_filename))
    #     blob = bucket.blob(zip_filename)
    #     blob.upload_from_filename(source_dir + "/" + zip_filename)
    # else:
    shutil.copy(source_dir + "/" + zip_filename, target_dir + "/" + zip_filename)


def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    data = data[~np.isnan(data)]
    a = 1.0 * data
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


def reduce_df_mem(df, without_columns = []):
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype
        if col in without_columns:
          next
        if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)                    
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                    elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                        df[col] = df[col].astype(np.uint64)
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df