import luigi
import pandas as pd
import numpy as np
import os
from mars_gym.data.task import BasePrepareDataFrames
from mars_gym.data.utils import DownloadDataset
import random

OUTPUT_PATH: str = os.environ[
    "OUTPUT_PATH"
] if "OUTPUT_PATH" in os.environ else os.path.join("output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "yoochoose")
DATASET_DIR: str = os.path.join(OUTPUT_PATH, "yoochoose", "dataset")


class PrepareDataset(luigi.Task):
    history_window: int = luigi.IntParameter(default=3)
    size_available_list: int = luigi.IntParameter(default=20)

    def requires(self):
        return DownloadDataset(dataset="yoochoose", output_path=OUTPUT_PATH)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "dataset_prepared.csv",))

    def filter(self, df):
        # Filter interactions

        df["Step"] = 1
        df["Step"] = df.groupby(["SessionID"])["Step"].apply(lambda x: x.cumsum())

        # Filters only interactions that have history
        df = df[df.Step > 1]

        # Filters only buy interactions
        df = df[df.Quantity > 0]

        return df

    def add_history_buys(self, df):
        df_list = []
        for g, df_g in df.groupby("SessionID"):
            list_of_indexes = []
            df_g["ItemID"].rolling(self.history_window, min_periods=1).apply(
                (lambda x: list_of_indexes.append(x.astype(int).tolist()) or 0),
                raw=False,
            )
            df_g["ItemID_history"] = list_of_indexes
            df_g["ItemID_history"] = df_g["ItemID_history"].shift(1)

            # add padding
            Item_prev = (
                df_g["ItemID_history"]
                .apply(lambda x: [] if x is np.nan else x)
                .apply(
                    lambda x: ([""] * (self.history_window - len(x)) + x)[
                        : self.history_window
                    ]
                )
            )
            df_list.append(Item_prev)

        df["ItemID_history"] = pd.concat(df_list)

        return df

    def add_available_items(self, df):
        all_items = list(df["ItemID"])
        df["available_items"] = df["ItemID"].apply(
            lambda item_id: random.sample(all_items, self.size_available_list - 1)
            + [item_id]
        )
        return df

    def add_info_timestamp(self, df, datetime_column="Timestamp"):
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df[datetime_column + "_dayofweek"] = df[datetime_column].dt.dayofweek

        return df

    def run(self):
        os.makedirs(DATASET_DIR, exist_ok=True)
        df = (
            pd.read_csv(self.input()[0].path, parse_dates=[1], header=None)
            .dropna()
            .sort_values([1])
        )
        df.columns = ["SessionID", "Timestamp", "ItemID", "Price", "Quantity"]
        df["ItemID"] = df["ItemID"].astype(str)
        df["SessionID"] = df["SessionID"].astype(str)

        # Add history information
        df = self.add_history_buys(df)

        # Filter Sessions
        df = self.filter(df)

        # Add timestamp information
        df = self.add_info_timestamp(df)

        # Add list of available items per interaction
        df = self.add_available_items(df)

        df.to_csv(self.output().path, index="Timestamp")


class PrepareInteractionDataFrame(BasePrepareDataFrames):
    history_window: int = luigi.IntParameter(default=5)
    size_available_list: int = luigi.IntParameter(default=20)

    def requires(self):
        return PrepareDataset(
            history_window=self.history_window,
            size_available_list=self.size_available_list,
        )

    @property
    def timestamp_property(self) -> str:
        return "Timestamp"

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    def read_data_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.input().path)

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        df["buy"] = (df["Quantity"] > 0).astype(int)

        return df


#################################################
class LoadAndPrepareDataset(luigi.Task):
    def requires(self):
        return DownloadDataset(dataset="processed_yoochoose", output_path=OUTPUT_PATH)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "processed_dataset.csv",))

    def add_info_timestamp(self, df, datetime_column="Timestamp"):
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df[datetime_column + "_dayofweek"] = df[datetime_column].dt.dayofweek

        return df

    def run(self):
        os.makedirs(DATASET_DIR, exist_ok=True)
        df = pd.read_csv(self.input()[0].path, parse_dates=[0])
        df["ItemID"] = df["ItemID"].astype(str)
        df["SessionID"] = df["SessionID"].astype(str)

        # Add timestamp information
        df = self.add_info_timestamp(df)

        # Add target interaction
        df["buy"] = (df["Quantity"] > 0).astype(int)

        df.to_csv(self.output().path)


class InteractionDataFrame(BasePrepareDataFrames):
    def requires(self):
        return LoadAndPrepareDataset()

    @property
    def timestamp_property(self) -> str:
        return "Timestamp"

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def read_data_frame_path(self) -> pd.DataFrame:
        return self.input().path

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        return df
