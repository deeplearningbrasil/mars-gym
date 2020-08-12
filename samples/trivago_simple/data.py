import luigi
import pandas as pd
import numpy as np
import datetime
import os
from typing import List, Tuple, Dict, Optional
from mars_gym.data.task import (
    BasePySparkTask,
    BasePrepareDataFrames,
)
from mars_gym.data.utils import DownloadDataset

OUTPUT_PATH: str = os.environ[
    "OUTPUT_PATH"
] if "OUTPUT_PATH" in os.environ else os.path.join("output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "processed_trivago_rio")
DATASET_DIR: str = os.path.join(OUTPUT_PATH, "processed_trivago_rio", "dataset")


class PrepareInteractionData(luigi.Task):
    def requires(self):
        return DownloadDataset(dataset="processed_trivago_rio", output_path=OUTPUT_PATH)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "dataset.csv",))

    def run(self):
        os.makedirs(DATASET_DIR, exist_ok=True)

        df = pd.read_csv(self.input()[0].path)
        df['impressions'] = df['impressions'].apply(lambda x: [] if x is np.nan else x)
        
        # .... transform dataset
        df.to_csv(self.output().path, index=False)


class PrepareMetaData(luigi.Task):
    def requires(self):
        return DownloadDataset(dataset="processed_trivago_rio", output_path=OUTPUT_PATH)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "metadata.csv",))

    def run(self):
        os.makedirs(DATASET_DIR, exist_ok=True)

        df = pd.read_csv(self.input()[1].path)
        # .... transform dataset

        # Save metadata processed
        df.to_csv(self.output().path, index="item_id")


class PrepareTrivagoDataFrame(BasePrepareDataFrames):
    def requires(self):
        return (
            PrepareInteractionData(),
            PrepareMetaData(),
        )

    @property
    def timestamp_property(self) -> str:
        return "timestamp"

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def read_data_frame_path(self) -> pd.DataFrame:
        return self.input()[0].path

    @property
    def metadata_data_frame_path(self) -> Optional[str]:
        return self.input()[1].path
