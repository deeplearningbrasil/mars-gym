import abc
import luigi

import pandas as pd
import numpy as np
import os
from mars_gym.data.task import BasePrepareDataFrames
from mars_gym.utils import files
from mars_gym.data.utils import DownloadDataset


class UnitTestDataFrames(BasePrepareDataFrames):
    def requires(self):
        return DownloadDataset(dataset="random")

    @property
    def timestamp_property(self) -> str:
        return "timestamp"

    @property
    def stratification_property(self) -> str:
        return "reward"

    @property
    def dataset_dir(self) -> str:
        return os.path.join(files.OUTPUT_PATH, "dataset")

    def data_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.input()[0].path)

    def read_data_frame(self) -> pd.DataFrame:
        return self.data_frame()

    def metadata_data_frame(self) -> pd.DataFrame:
        return None

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        df["n_users"] = len(self.data_frame().user.unique())
        df["n_items"] = len(self.data_frame().item.unique())

        return df
