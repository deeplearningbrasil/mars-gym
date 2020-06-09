import pandas as pd
import numpy as np
import datetime
import os
from mars_gym.utils.utils import random_date
from mars_gym.data.task import BasePrepareDataFrames


class RandomData:
    def __init__(self, n_users: int = 10, n_items: int = 10, size=1000, seed: int = 42):
        self.n_users = n_users
        self.n_items = n_items
        self.size = size
        self.seed = seed
        np.random.seed(seed)

    def data(self):
        users = np.random.randint(self.n_users, size=self.size)
        items = np.random.randint(self.n_items, size=self.size)
        start_date = datetime.datetime(2013, 9, 20, 13, 00)
        timestamp = [
            x.strftime("%d/%m/%y %H:%M") for x in random_date(start_date, self.size)
        ]

        reward = np.random.randint(2, size=self.size)

        df = pd.DataFrame(
            {"user": users, "item": items, "timestamp": timestamp, "reward": reward}
        )

        return df


class UnitTestDataFrames(BasePrepareDataFrames):
    def requires(self):
        return []

    @property
    def timestamp_property(self) -> str:
        return "timestamp"

    @property
    def stratification_property(self) -> str:
        return "rewards"

    @property
    def dataset_dir(self) -> str:
        return os.path.join("tests", "output", "test")

    def data_frame(self) -> pd.DataFrame:
        return RandomData().data()

    def read_data_frame(self) -> pd.DataFrame:
        return self.data_frame()

    def metadata_data_frame(self) -> pd.DataFrame:
        return None

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        df["n_users"] = len(self.data_frame().user.unique())
        df["n_items"] = len(self.data_frame().item.unique())

        return df
