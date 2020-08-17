import luigi
import pandas as pd
import numpy as np
import datetime
import os
import random
from typing import List, Tuple, Dict, Optional
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from mars_gym.data.task import (
    BasePySparkTask,
    BasePrepareDataFrames,
)
from pyspark.sql.window import Window

from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType
from pyspark.sql.functions import (
    when,
    collect_list,
    lit,
    col,
    max,
)
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from mars_gym.data.utils import DownloadDataset
from mars_gym.utils.utils import _pad_sequence, to_array, array_index_udf
from pyspark.sql.types import ArrayType, FloatType


OUTPUT_PATH: str = os.environ[
    "OUTPUT_PATH"
] if "OUTPUT_PATH" in os.environ else os.path.join("output")
BASE_DIR: str = os.path.join(OUTPUT_PATH, "trivago_rio")
DATASET_DIR: str = os.path.join(OUTPUT_PATH, "trivago_rio", "dataset")


class PrepareMetaData(luigi.Task):
    def requires(self):
        return DownloadDataset(dataset="trivago_rio", output_path=OUTPUT_PATH)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(DATASET_DIR, "item_metadata_transform.csv",)
        )

    # 'a|b|c' -> ['a', 'b', 'c']
    #
    #
    def split_df_columns(self, df, column):
        tf = CountVectorizer(tokenizer=lambda x: x.split("|"))
        tf_df = tf.fit_transform(df[column]).todense()
        tf_df = pd.DataFrame(tf_df, columns=sorted(tf.vocabulary_.keys()))
        return tf_df.astype("uint8")

    def run(self):
        os.makedirs(DATASET_DIR, exist_ok=True)

        df_meta = pd.read_csv(self.input()[1].path)
        # print(df_meta)
        # Split feature columns
        tf_prop_meta = self.split_df_columns(df_meta, "properties")
        df_meta = df_meta.join(tf_prop_meta).drop(["properties"], axis=1).astype(int)

        df_meta["list_metadata"] = df_meta.drop("item_id", 1).values.tolist()

        # Save metadata processed
        df_meta.to_csv(self.output().path, index="item_id")


class PrepareHistoryInteractionData(BasePySparkTask):
    window_hist: int = luigi.IntParameter(default=5)

    def requires(self):
        return (
            DownloadDataset(dataset="trivago_rio", output_path=OUTPUT_PATH),
            PrepareMetaData(),
        )

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "interaction_dataset.csv",))

    def filter_data(self, spark, df):
        df_meta = spark.read.csv(self.input()[1].path, header=True, inferSchema=True)

        df = df.filter(df.action_type.isin(["interaction item info", "clickout item"]))
        df = df.dropna(subset=["reference"])
        df = df.join(df_meta.select("item_id"), df.reference == df_meta.item_id)
        df = df.select(
            "session_id",
            "user_id",
            "timestamp",
            "action_type",
            "reference",
            "impressions",
        )

        return df

    def group_history_data(self, df):
        # Build window history
        win_over_session = (
            Window.partitionBy("session_id")
            .orderBy("timestamp")
            .rangeBetween(Window.unboundedPreceding, -1)
        )

        # Apply udf function for padding array
        pad_fix_length = F.udf(
            lambda arr: arr[: self.window_hist]
            + [""] * (self.window_hist - len(arr[: self.window_hist])),
            ArrayType(StringType()),
        )

        #
        df = df.withColumn(
            "list_reference_item",
            pad_fix_length(collect_list("reference").over(win_over_session)),
        )

        print(df.show(5))

        return df

    def transform_data(self, df):
        df = df.withColumn("impressions", F.split(df.impressions, "\|"))

        df = df.withColumnRenamed("reference", "item_id")

        df = df.withColumn("pos_item_id", array_index_udf(df.impressions, df.item_id))

        df = df.withColumn(
            "clicked", when(df.action_type == "clickout item", 1.0).otherwise(0.0),
        )

        return df

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)
        spark = SparkSession(sc)

        df = spark.read.csv(
            self.input()[0][0].path, header=True, inferSchema=True
        ).orderBy("timestamp")

        df = self.filter_data(spark, df)
        df = self.group_history_data(df)
        df = self.transform_data(df)

        df = df.cache().toPandas()
        
        print("=========================")
        print(df)

        df['impressions'] = list(df.apply(lambda row: sum([[row.item_id], random.sample(list(df['item_id']), 24)], []), axis=1))
        print(df.head())
        df.to_csv(self.output().path, index=False)


class PrepareTrivagoDataFrame(BasePrepareDataFrames):
    window_hist: int = luigi.IntParameter(default=5)

    def requires(self):
        return (
            PrepareHistoryInteractionData(window_hist=self.window_hist),
            PrepareMetaData(),
        )

    @property
    def timestamp_property(self) -> str:
        return "timestamp"

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def metadata_data_frame_path(self) -> Optional[str]:
        return self.input()[1].path

    @property
    def read_data_frame_path(self) -> pd.DataFrame:
        return self.input()[0].path

    def transform_data_frame(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        return df
