import os

import luigi
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, collect_set

from recommendation.task.data_preparation.base import BasePrepareDataFrames, BasePySparkTask

BASE_DIR: str = os.path.join("output", "new_ifood")


class BaseDir(luigi.Config):
    dataset_processed_path: str = luigi.Parameter(default=os.path.join(BASE_DIR, "dataset"), is_global=True)
    dataset_raw_path: str = luigi.Parameter(default=os.path.join(BASE_DIR, "new_ifood_dataset"), is_global=True)

    @property
    def dataset_processed(self):
        if "DATASET_PROCESSED_PATH" in os.environ:
            return os.environ["DATASET_PROCESSED_PATH"]
        else:
            return self.dataset_processed_path


class CheckDataset(luigi.Task):
    def output(self):
        return luigi.LocalTarget(os.path.join(BaseDir().dataset_raw_path, "train")), \
               luigi.LocalTarget(os.path.join(BaseDir().dataset_raw_path, "validation")), \
               luigi.LocalTarget(os.path.join(BaseDir().dataset_raw_path, "test")), \
               luigi.LocalTarget(os.path.join(BaseDir().dataset_raw_path, "index", "user",
                                              "part-00000-tid-1042389914227284358-91f56a10-24d8-4061-90a5-fc2770d93e62-682264-1-c000.csv")), \
               luigi.LocalTarget(os.path.join(BaseDir().dataset_raw_path, "index", "merchants",
                                              "part-00000-tid-8169454250339942259-802d73b2-083a-4435-b1eb-e7964ad5deb6-684999-1-c000.csv"))

    def run(self):
        raise AssertionError(
            f"As seguintes pastas são esperadas com o dataset: {[output.path for output in self.output()]}")

class IndexTrainDataset(BasePySparkTask):
    def requires(self):
        return CheckDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join(BaseDir().dataset_processed_path, "indexed_orders_train_data.parquet"))

    def main(self, sc: SparkContext, *args):
        os.makedirs(BaseDir().dataset_processed_path, exist_ok=True)

        spark = SparkSession(sc)

        orders_df   = spark.read.parquet(self.input()[0].path)
        account_df  = spark.read.csv(self.input()[3].path, header=True, inferSchema=True)
        merchant_df = spark.read.csv(self.input()[4].path, header=True, inferSchema=True)

        orders_df = orders_df\
            .join(account_df, ["account_idx"], how="inner")\
            .join(merchant_df, ["merchant_idx"], how="inner")

        orders_df.write.parquet(self.output().path)


class PrepareNewIfoodIndexedOrdersTestData(BasePySparkTask):
    def requires(self):
        return CheckDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join(BaseDir().dataset_processed_path, "indexed_orders_test_data.parquet"))

    def main(self, sc: SparkContext, *args):
        os.makedirs(BaseDir().dataset_processed_path, exist_ok=True)

        spark = SparkSession(sc)

        orders_df = spark.read.parquet(self.input()[1].path)
        account_df = spark.read.csv(self.input()[3].path, header=True, inferSchema=True)
        merchant_df = spark.read.csv(self.input()[4].path, header=True, inferSchema=True)

        orders_df = orders_df.select("order_id", "account_id", "merchant_id", "merc_list")\
            .join(account_df, orders_df.account_id == account_df.account_id, how="inner")\
            .join(merchant_df, orders_df.merchant_id == merchant_df.merchant_id, how="inner")\
            .select("order_id", "account_idx", "merchant_idx", "merc_list")\
            .withColumn("merchant_id_from_list", explode(orders_df.merc_list))\
            .drop("merc_list")

        merchant_for_list_df = merchant_df.select(col("merchant_id").alias("merchant_id_for_list"), col("merchant_idx").alias("merchant_idx_for_list"))
        orders_df = orders_df.join(merchant_for_list_df, orders_df.merchant_id_from_list == merchant_for_list_df.merchant_id_for_list, how="inner") \
            .groupBy(["order_id", "account_idx", "merchant_idx"])\
            .agg(collect_set("merchant_idx_for_list").alias("merchant_idx_list"))\
            .select("order_id", "account_idx", "merchant_idx", "merchant_idx_list")

        orders_df.write.parquet(self.output().path)


class ListAccountMerchantTuplesForNewIfoodIndexedOrdersTestData(BasePySparkTask):
    def requires(self):
        return PrepareNewIfoodIndexedOrdersTestData()

    def output(self):
        return luigi.LocalTarget(os.path.join(BaseDir().dataset_processed_path, "account_merchant_tuples_from_test_data.parquet"))

    def main(self, sc: SparkContext, *args):
        os.makedirs(BaseDir().dataset_processed_path, exist_ok=True)

        spark = SparkSession(sc)

        df = spark.read.parquet(self.input().path)

        tuples_df = df.select("account_idx", "merchant_idx")

        tuples_df = tuples_df.union(df.select("account_idx", "merchant_idx_list")
                                    .withColumn("merchant_idx", explode(df.merchant_idx_list))
                                    .select("account_idx", "merchant_idx")).dropDuplicates()

        tuples_df.write.parquet(self.output().path)

import numpy as np
class IndexDataset(luigi.Task):
    def requires(self):
        return CheckDataset()

    @property
    def dataset_dir(self) -> str:
        return BaseDir().dataset_processed_path

    def output(self):
        return luigi.LocalTarget(os.path.join(BaseDir().dataset_processed_path, "indexed_orders_valid_data.csv"))

    def add_merc_list_index(self, df, item_df):
        item_dict = item_df.set_index("merchant_id").to_dict()
        df['merc_list_idx'] = df['merc_list'].apply(lambda l:  [item_dict['merchant_idx'][i] for i in l if i in item_dict['merchant_idx']])

        df = df[df.merc_list_idx.apply(lambda x: len(x)) > 10]

        return df

    def index_shift(self, df):

        order = {'weekday breakfast': 1,
                'weekday dawn': 0,
                'weekday dinner': 4,
                'weekday lunch': 2,
                'weekday snack': 3,
                'weekend dawn': 5,
                'weekend dinner': 7,
                'weekend lunch': 6}
        df = df.join(pd.get_dummies(df['shift']))
        df['shift'] = df['shift'].map(order)

        return df

    def run(self):
        os.makedirs(self.dataset_dir, exist_ok=True)

        df   = pd.read_parquet(self.input()[1].path)
        user_df  = pd.read_csv(self.input()[3].path)
        item_df  = pd.read_csv(self.input()[4].path)
        #
        df   = df.merge(user_df, on="account_id").merge(item_df, on="merchant_id")

        df   = self.add_merc_list_index(df, item_df)
        df   = self.index_shift(df)
        
        df['buys'] = 1
        df['merchant_buys_cum'] = df.groupby("merchant_idx")["buys"].transform("cumsum")
        df['account_buys_cum']  = df.groupby("account_idx")["buys"].transform("cumsum")

        df['avg_merc_score']   = df['merc_score'].apply(np.mean)
        df['avg_delivery_fee'] = df['delivery_fee'].apply(np.mean)
        df['avg_distance']     = df['distance'].apply(np.mean)

        df = df.sort_values(['order_date_local', 'shift'])
        
        df.to_csv(self.output().path, index=False)
        
class BasePrepareNewIfoodDataFrames(BasePrepareDataFrames):
    @property
    def dataset_dir(self) -> str:
        return BaseDir().dataset_processed_path

    @property
    def read_data_frame(self) -> pd.DataFrame:
        return None

    @property
    def stratification_property(self) -> str:
        return None

    @property
    def num_users(self):
        if not hasattr(self, "_num_users"):
            accounts_df = pd.read_csv(self.input()[0][3].path)
            self._num_users = len(accounts_df)
        return self._num_users

    @property
    def num_businesses(self):
        if not hasattr(self, "_num_businesses"):
            merchants_df = pd.read_csv(self.input()[0][4].path)
            self._num_businesses = len(merchants_df)
        return self._num_businesses

    def requires(self):
        return CheckDataset(), IndexDataset()

    def run(self):
        os.makedirs(self.dataset_dir, exist_ok=True)

        train_df = pd.read_parquet(self.input()[0][0].path)
        test_df  = pd.read_parquet(self.input()[0][2].path)
        val_df   = pd.read_csv(self.input()[1].path)

        user_df   = pd.read_csv(self.input()[0][3].path)
        item_df   = pd.read_csv(self.input()[0][4].path)

        train_df  = train_df.merge(user_df, on="account_idx").merge(item_df, on="merchant_idx")
        test_df   = test_df.merge(user_df, on="account_idx").merge(item_df, on="merchant_idx")

        for field, value in self.eq_filters.items():
            train_df = train_df[train_df[field] == value]
            test_df = test_df[test_df[field] == value]

        for field, value in self.neq_filters.items():
            train_df = train_df[train_df[field] != value]
            test_df = test_df[test_df[field] != value]

        for field, value in self.isin_filters.items():
            train_df = train_df[train_df[field].isin(value)]
            test_df = test_df[test_df[field].isin(value)]

        self.transform_data_frame(train_df).to_csv(self.output()[0].path, index=False)
        self.transform_data_frame(val_df).to_csv(self.output()[1].path, index=False)
        self.transform_data_frame(test_df).to_csv(self.output()[2].path, index=False)

    def transform_data_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df["n_users"] = self.num_users
        df["n_items"] = self.num_businesses
        return df

class PrepareNewIfoodInteractionsDataFrames(BasePrepareNewIfoodDataFrames):
    def transform_data_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().transform_data_frame(df)
        df["binary_buys"] = (df["buys"] > 0).astype(float)
        df["buys"]        = df["buys"].astype(float)

        return df
