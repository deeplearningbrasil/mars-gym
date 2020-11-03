import abc
import itertools
import math
import os
from typing import List, Tuple, Dict, Optional
import re
import tempfile
import luigi
import numpy as np
import pandas as pd
import psutil
import inspect
import shutil

import requests
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import random
from luigi.contrib.spark import PySparkTask
from pyspark import SparkConf
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
import mars_gym


class BaseDownloadDataset(luigi.Task, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def output(self) -> luigi.LocalTarget:
        pass

    @property
    @abc.abstractmethod
    def url(self) -> str:
        pass

    def run(self):
        # Streaming, so we can iterate over the response.
        r = requests.get(self.url, stream=True)
        output_path = self.output().path

        # Total size in bytes.
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        wrote = 0
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        with open(output_path, "wb") as f:
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


class BasePrepareDataFrames(luigi.Task, metaclass=abc.ABCMeta):
    session_test_size: float = luigi.FloatParameter(default=0.10)
    test_size: float = luigi.FloatParameter(default=0.2)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)
    dataset_split_method: str = luigi.ChoiceParameter(
        choices=["holdout", "time", "column", "k_fold"], default="time"
    )
    column_stratification: str = luigi.Parameter(default=None)
    test_split_type: str = luigi.ChoiceParameter(
        choices=["random", "time"], default="random"
    )
    item_column: str = luigi.Parameter()
    available_arms_column_name: str = luigi.Parameter(default='available_arms')
    n_splits: int = luigi.IntParameter(default=10)
    split_index: int = luigi.IntParameter(default=0)
    val_size: float = luigi.FloatParameter(default=0.2)
    sampling_strategy: str = luigi.ChoiceParameter(
        choices=["oversample", "undersample", "none"], default="none"
    )
    balance_fields: List[str] = luigi.ListParameter(default=[])
    sampling_proportions: Dict[str, Dict[str, float]] = luigi.DictParameter(default={})
    use_sampling_in_validation: bool = luigi.BoolParameter(default=False)
    eq_filters: Dict[str, any] = luigi.DictParameter(default={})
    neq_filters: Dict[str, any] = luigi.DictParameter(default={})
    isin_filters: Dict[str, any] = luigi.DictParameter(default={})
    seed: int = luigi.IntParameter(default=42)

    VALIDATION_DATA = "VALIDATION_DATA"
    TRAIN_DATA = "TRAIN_DATA"
    TEST_GENERATOR = "TEST_GENERATOR"

    @property
    @abc.abstractmethod
    def dataset_dir(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def read_data_frame_path(self) -> Optional[str]:
        pass

    def read_data_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.read_data_frame_path)

    @property
    def stratification_property(self) -> str:
        pass

    @property
    def timestamp_property(self) -> Optional[str]:
        return None

    @property
    def metadata_data_frame_path(self) -> Optional[str]:
        return None

    def output(self) -> Tuple[luigi.LocalTarget, ...]:
        task_hash = self.task_id
        if self.dataset_split_method == "k_fold":
            output = (
                luigi.LocalTarget(
                    os.path.join(
                        self.dataset_dir,
                        "train_[%dof%d]_test=%s_%d_%s_%s.csv"
                        % (
                            self.split_index + 1,
                            self.n_splits,
                            self.test_split_type,
                            self.seed,
                            self.sampling_strategy,
                            task_hash,
                        ),
                    )
                ),
                luigi.LocalTarget(
                    os.path.join(
                        self.dataset_dir,
                        "val_[%of%d]_test=%s_%d_%s.csv"
                        % (
                            self.split_index + 1,
                            self.n_splits,
                            self.test_split_type,
                            self.seed,
                            task_hash,
                        ),
                    )
                ),
                luigi.LocalTarget(
                    os.path.join(
                        self.dataset_dir,
                        "test_%.2f_test=%s_%d_%s.csv"
                        % (self.test_size, self.test_split_type, self.seed, task_hash),
                    )
                ),
            )

        else:
            output = (
                luigi.LocalTarget(
                    os.path.join(
                        self.dataset_dir,
                        "train_%.2f_test=%s_%d_%s_%s.csv"
                        % (
                            self.val_size,
                            self.test_split_type,
                            self.seed,
                            self.sampling_strategy,
                            task_hash,
                        ),
                    )
                ),
                luigi.LocalTarget(
                    os.path.join(
                        self.dataset_dir,
                        "val_%.2f_test=%s_%d_%s.csv"
                        % (self.val_size, self.test_split_type, self.seed, task_hash),
                    )
                ),
                luigi.LocalTarget(
                    os.path.join(
                        self.dataset_dir,
                        "test_%.2f_test=%s_%d_%s.csv"
                        % (self.test_size, self.test_split_type, self.seed, task_hash),
                    )
                ),
            )
        if self.metadata_data_frame_path:
            return output + (luigi.LocalTarget(self.metadata_data_frame_path),)

        return output

    def run(self):
        os.makedirs(self.dataset_dir, exist_ok=True)

        df = self.read_data_frame()

        for field, value in self.eq_filters.items():
            df = df[df[field] == value]

        for field, value in self.neq_filters.items():
            df = df[df[field] != value]

        for field, value in self.isin_filters.items():
            df = df[df[field].isin(value)]
        
        self.create_available_arms(df)
        
        self.train_df, self.val_df, self.test_df = self.split_dataset(df)

        self.transform_data_frame(self.train_df, data_key=self.TRAIN_DATA).to_csv(
            self.output()[0].path, index=False
        )
        self.transform_data_frame(self.val_df, data_key=self.VALIDATION_DATA).to_csv(
            self.output()[1].path, index=False
        )
        self.transform_data_frame(self.test_df, data_key=self.TEST_GENERATOR).to_csv(
            self.output()[2].path, index=False
        )

    def split_dataset(self, df):
        if self.test_size:
            if self.test_split_type == "random":
                train_df, test_df = self.random_train_test_split(
                    df, test_size=self.test_size
                )
            else:
                train_df, test_df = self.time_train_test_split(
                    df, test_size=self.test_size
                )
        else:
            train_df, test_df = df, df[:0]
        
        if self.val_size:
            if self.dataset_split_method == "holdout":
                train_df, val_df = self.random_train_test_split(
                    train_df, test_size=self.val_size
                )
            elif self.dataset_split_method == "column":
                train_df, val_df = self.column_train_test_split(
                    train_df, test_size=self.val_size
                )
            elif self.dataset_split_method == "time":
                train_df, val_df = self.time_train_test_split(
                    train_df, test_size=self.val_size
                )
            else:
                train_df, val_df = self.kfold_split(train_df)
        else:
            train_df, val_df = train_df, train_df[:0]

        if self.sampling_strategy != "none":
            train_df = self.balance_dataset(train_df)
            if self.use_sampling_in_validation:
                val_df = self.balance_dataset(val_df)

        return train_df, val_df, test_df

    def create_available_arms(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
        # if self.available_arms_column_name not in df.columns:

        #     def add_arms(arm, item_unique):
        #         arms = random.sample(item_unique, min(99, len(item_unique)))
        #         arms.append(arm)
        #         arms = list(np.unique(arms))   
        #         return arms       
        #     item_unique = list(df[self.item_column].drop_duplicates().values)
        #     df[self.available_arms_column_name] = df.apply(lambda row: add_arms(row[self.item_column], item_unique), axis=1)

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        return df

    def kfold_split(self, df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )
        train_indices, val_indices = next(
            itertools.islice(
                skf.split(
                    df,
                    df[self.stratification_property]
                    if self.stratification_property
                    else None,
                ),
                self.split_index,
                self.split_index + 1,
            )
        )
        train_df, test_df = df.iloc[train_indices], df.iloc[val_indices]
        return train_df, test_df

    def column_train_test_split(
        self, df: pd.DataFrame, test_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        keys = df[self.column_stratification].unique()
        keys_train, keys_test = train_test_split(
            keys, test_size=test_size, random_state=self.seed
        )
        return (
            df[df[self.column_stratification].isin(keys_train)],
            df[df[self.column_stratification].isin(keys_test)],
        )

    def time_train_test_split(
        self, df: pd.DataFrame, test_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.timestamp_property:
            df = df.sort_values(self.timestamp_property)
        size = len(df)
        cut = int(size - size * test_size)

        return df.iloc[:cut], df.iloc[cut:]

    def random_train_test_split(
        self, df: pd.DataFrame, test_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(
            df,
            test_size=test_size,
            stratify=df[self.stratification_property]
            if self.stratification_property
            else None,
            random_state=self.seed,
        )

    def _create_sampling_strategy(self, df: pd.DataFrame, balance_field: str):
        if balance_field in self.sampling_proportions:
            field_unique_values, counts = np.unique(
                df[balance_field], return_counts=True
            )
            if self.sampling_strategy == "oversample":
                min_num_samples = max(counts)
                min_proportion = min(self.sampling_proportions[balance_field].values())
                return {
                    field_value: int(
                        self.sampling_proportions[balance_field][field_value]
                        * min_num_samples
                        / min_proportion
                    )
                    for field_value in field_unique_values
                }
            elif self.sampling_strategy == "undersample":
                max_num_samples = min(counts)
                max_proportion = max(self.sampling_proportions[balance_field].values())
                return {
                    field_value: int(
                        self.sampling_proportions[balance_field][field_value]
                        * max_num_samples
                        / max_proportion
                    )
                    for field_value in field_unique_values
                }
        return "auto"

    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        sampler = dict(oversample=RandomOverSampler, undersample=RandomUnderSampler)

        random_sampler_cls = sampler.get(self.sampling_strategy)

        if random_sampler_cls is None:
            return df
        index_resampled = df.index
        for balance_field in self.balance_fields:
            random_sampler = random_sampler_cls(
                sampling_strategy=self._create_sampling_strategy(df, balance_field),
                random_state=self.seed,
            )
            index_resampled, _ = random_sampler.fit_sample(
                np.array(index_resampled).reshape(-1, 1),
                df.loc[index_resampled][balance_field],
            )
            index_resampled = index_resampled.flatten()

        return df.loc[index_resampled]


class BasePySparkTask(PySparkTask):
    def setup(self, conf: SparkConf):
        conf.set("spark.local.dir", os.path.join("output", "spark"))
        conf.set("spark.driver.maxResultSize", self._get_available_memory())
        # conf.set("spark.sql.shuffle.partitions", os.cpu_count())
        # conf.set("spark.default.parallelism", os.cpu_count())

    @property
    def driver_memory(self):
        return self._get_available_memory()

    def _get_available_memory(self) -> str:
        return f"{int(psutil.virtual_memory().available / (1024 * 1024 * 1024))}g"

    # FIX https://github.com/spotify/luigi/pull/2502/files
    def run(self):
        path_name_fragment = re.sub(r"[^\w]", "_", self.name)
        self.run_path = tempfile.mkdtemp(prefix=path_name_fragment)
        self.run_pickle = os.path.join(
            self.run_path, ".".join([path_name_fragment, "pickle"])
        )
        with open(self.run_pickle, "wb") as fd:
            # Copy module file to run path.
            module_file_path = os.path.abspath(inspect.getfile(self.__class__))
            base_module = self.__class__.__module__.split(".")[0]
            module_folder_path = module_file_path[
                : module_file_path.find(base_module) + len(base_module)
            ]
            # from IPython import embed; embed()

            # Copy MARS
            # mars_module_folder_path = os.path.join(os.path.abspath(inspect.getfile(mars_gym))[:-11], 'mars_gym')
            shutil.copytree(
                os.path.abspath(inspect.getfile(mars_gym))[:-11],
                os.path.join(self.run_path, "mars_gym"),
            )

            shutil.copytree(
                module_folder_path, os.path.join(self.run_path, base_module)
            )
            shutil.copy(module_file_path, os.path.join(self.run_path, "."))

            self._dump(fd)
        try:
            super(PySparkTask, self).run()
        finally:
            shutil.rmtree(self.run_path)
