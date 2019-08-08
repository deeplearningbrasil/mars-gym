import os

import luigi
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import create_map, udf
from pyspark.sql.types import IntegerType

from recommendation.task.data_preparation.base import BasePySparkTask, BasePrepareDataFrames

BASE_DIR: str = os.path.join("output", "ifood")
DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")


class CheckInteractionsDataset(luigi.Task):
    def output(self):
        return luigi.LocalTarget(os.path.join(BASE_DIR, "interactions_training_data")), \
               luigi.LocalTarget(os.path.join(BASE_DIR, "interactions_test_data"))

    def run(self):
        raise AssertionError(
            f"As seguintes pastas são esperadas com o dataset: '{self.output()[0].path}' e '{self.output()[1].path}'")


class GenerateIndicesForAccountsAndMerchantsOfInteractionsDataset(BasePySparkTask):
    def requires(self):
        return CheckInteractionsDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "accounts_indices_for_interactions_data.csv")), \
               luigi.LocalTarget(os.path.join(DATASET_DIR, "merchants_indices_for_interactions_data.csv"))

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        spark = SparkSession(sc)

        train_df = spark.read.parquet(self.input()[0].path)

        account_df = train_df.select("account_id").distinct()
        merchant_df = train_df.select("merchant_id").distinct()

        account_df.toPandas().to_csv(self.output()[0].path, index_label="account_idx")
        merchant_df.toPandas().to_csv(self.output()[1].path, index_label="merchant_idx")


class IndexAccountsAndMerchantsOfInteractionsDataset(BasePySparkTask):

    def requires(self):
        return CheckInteractionsDataset(), GenerateIndicesForAccountsAndMerchantsOfInteractionsDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "indexed_interactions_data.csv"))

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        spark = SparkSession(sc)

        train_df = spark.read.parquet(self.input()[0][0].path)
        account_df = spark.read.csv(self.input()[1][0].path, header=True)
        merchant_df = spark.read.csv(self.input()[1][1].path, header=True)

        train_df = train_df.join(account_df, train_df.account_id == account_df.account_id)
        train_df = train_df.join(merchant_df, train_df.merchant_id == merchant_df.merchant_id)

        train_df = train_df.select("account_idx", "merchant_idx", "visits", "buys")

        train_df.toPandas().to_csv(self.output().path, index=False)


class PrepareIfoodBinaryBuysInteractionsDataFrames(BasePrepareDataFrames):

    def requires(self):
        return GenerateIndicesForAccountsAndMerchantsOfInteractionsDataset(),\
               IndexAccountsAndMerchantsOfInteractionsDataset()

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def stratification_property(self) -> str:
        return "buys"

    @property
    def num_users(self):
        if not hasattr(self, "_num_users"):
            accounts_df = pd.read_csv(self.input()[0][0].path)
            self._num_users = len(accounts_df)
        return self._num_users

    @property
    def num_businesses(self):
        if not hasattr(self, "_num_businesses"):
            merchants_df = pd.read_csv(self.input()[0][1].path)
            self._num_businesses = len(merchants_df)
        return self._num_businesses

    def read_data_frame(self) -> pd.DataFrame:
        df = pd.read_csv(self.input()[1].path)
        df["buys"] = (df["buys"] > 0).astype(float)
        df["n_users"] = self.num_users
        df["n_items"] = self.num_businesses

        return df
