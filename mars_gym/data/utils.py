import abc
import luigi

import pandas as pd
import numpy as np
import datetime
import os
from mars_gym.utils.utils import random_date
from mars_gym.data.task import BasePrepareDataFrames
from urllib.request import urlopen, urlretrieve
from mars_gym.utils import files
import requests
from tqdm import tqdm
import math
from mars_gym.utils import files

DATASETS = dict(
    random=["https://storage.googleapis.com/mars-gym-dataset/raw/random/dataset.csv"],
    yoochoose=[
        "https://storage.googleapis.com/mars-gym-dataset/raw/yoochoose/yoochoose-buys.dat"
    ],
    processed_yoochoose=[
        "https://storage.googleapis.com/mars-gym-dataset/process/yoochoose/dataset.csv"
    ],
    trivago_rio=[
        "https://storage.googleapis.com/mars-gym-dataset/raw/trivago/rio/train.csv",
        "https://storage.googleapis.com/mars-gym-dataset/raw/trivago/rio/item_metadata.csv",
    ],
    processed_trivago_rio=[
        "https://storage.googleapis.com/mars-gym-dataset/process/trivago/rio/interaction_dataset.csv",
        "https://storage.googleapis.com/mars-gym-dataset/process/trivago/rio/item_metadata_transform.csv",
    ],
)


def datasets():
    return list(DATASETS.keys())


def load_dataset(name, cache=True, output_path=".", **kws):
    results = []
    for url in DATASETS[name]:

        output_file = os.path.join(output_path, name, os.path.basename(url))
        if not os.path.isfile(output_file) or not cache:
            # Streaming, so we can iterate over the response.
            r = requests.get(url, stream=True)

            # Total size in bytes.
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024
            wrote = 0
            os.makedirs(os.path.split(output_file)[0], exist_ok=True)
            with open(output_file, "wb") as f:
                for data in tqdm(
                    r.iter_content(block_size),
                    total=math.ceil(total_size // block_size),
                    unit="KB",
                    unit_scale=True,
                ):
                    wrote = wrote + len(data)
                    f.write(data)
            if total_size != 0 and wrote != total_size:
                raise ConnectionError("ERROR, something went wrong")

        df = pd.read_csv(output_file, **kws)

        results.append(df)
    return results


class DownloadDataset(luigi.Task, metaclass=abc.ABCMeta):
    output_path: str = luigi.Parameter(default=files.OUTPUT_PATH)
    dataset: str = luigi.ChoiceParameter(choices=DATASETS.keys())

    def output(self):
        return [
            luigi.LocalTarget(
                os.path.join(self.output_path, self.dataset, os.path.basename(p))
            )
            for p in DATASETS[self.dataset]
        ]

    def run(self):
        load_dataset(self.dataset, output_path=self.output_path)
