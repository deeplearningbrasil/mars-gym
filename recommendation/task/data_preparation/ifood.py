import os

import luigi
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, collect_set
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import explode, first
import pyspark.sql.functions as f
from pyspark.sql.functions import col, expr, when
from pyspark.sql.types import DateType
import pyspark

from recommendation.task.data_preparation.base import BasePySparkTask, BasePrepareDataFrames

BASE_DIR: str = os.path.join("output", "ifood")
DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")

# df train 2019-03-26 at 2019-06-17
#
WINDOW_FILTER_DF = dict(one_week='2019-06-10', one_month='2019-05-17', all='2019-03-26')

class CheckInteractionsDataset(luigi.Task):
    def output(self):
        return luigi.LocalTarget(os.path.join(BASE_DIR, "interactions_training_data")), \
               luigi.LocalTarget(os.path.join(BASE_DIR, "interactions_test_data"))

    def run(self):
        raise AssertionError(
            f"As seguintes pastas são esperadas com o dataset: '{self.output()[0].path}' e '{self.output()[1].path}'")

class FilterDataset(BasePySparkTask):
    def requires(self):
        return CheckInteractionsDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join(BASE_DIR, "interactions_training_data_{}".format(self.window_filter))), \
               luigi.LocalTarget(os.path.join(BASE_DIR, "interactions_test_data"))

    def main(self, sc: SparkContext, *args):
        #os.makedirs(os.path.join(BASE_DIR, "interactions_training_data_{}".format(self.window_filter)), exist_ok=True)

        spark = SparkSession(sc)

        df    = spark.read.parquet(self.input()[0].path)
        print(df.printSchema())

        if self.window_filter != 'all':

            # Explode to Filter
            df_t = df.select(df.account_id, df.merchant_id, 
                             explode(df.visit_events).alias('visit_event'),
                             f.col('purchase_events')[0].alias('purchase_event'))
            df_t = df_t.withColumn("visit_event",df_t['visit_event'].cast(DateType()))
            df_t = df_t.withColumn('buys', when(df_t.purchase_event == df_t.visit_event, 1).otherwise(0)).persist( pyspark.StorageLevel.MEMORY_AND_DISK_2 )

            # Filter
            df_f = df_t.filter(f.col("visit_event") > f.lit(WINDOW_FILTER_DF[self.window_filter])).cache()

            # Implode
            df = df_f.groupby([df_f.account_id, df_f.merchant_id]).agg(
                f.count('visit_event').alias('visits'),
                f.collect_list('visit_event').alias('visit_events'),
                when(f.sum('buys') >= 1, 1).alias('buys'),
                when(f.size(f.collect_set('purchase_event')) == 0, f.lit(None)).otherwise(f.collect_set('purchase_event')).alias('purchase_events')
            ).cache()

        df.write.parquet(self.output()[0].path)

class GenerateIndicesForAccountsAndMerchantsOfInteractionsDataset(BasePySparkTask):
    def requires(self):
        return FilterDataset(window_filter=self.window_filter)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, self.window_filter, "accounts_indices_for_interactions_data.csv")), \
               luigi.LocalTarget(os.path.join(DATASET_DIR, self.window_filter, "merchants_indices_for_interactions_data.csv"))

    def main(self, sc: SparkContext, *args):
        os.makedirs(os.path.join(DATASET_DIR, self.window_filter), exist_ok=True)

        spark = SparkSession(sc)

        train_df = spark.read.parquet(self.input()[0].path)

        account_df = train_df.select("account_id").distinct()
        merchant_df = train_df.select("merchant_id").distinct()

        account_df.toPandas().to_csv(self.output()[0].path, index_label="account_idx")
        merchant_df.toPandas().to_csv(self.output()[1].path, index_label="merchant_idx")


class IndexAccountsAndMerchantsOfInteractionsDataset(BasePySparkTask):
    def requires(self):
        return FilterDataset(window_filter=self.window_filter), \
                    GenerateIndicesForAccountsAndMerchantsOfInteractionsDataset(window_filter=self.window_filter)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, self.window_filter, "indexed_interactions_data.csv"))

    def main(self, sc: SparkContext, *args):
        os.makedirs(os.path.join(DATASET_DIR, self.window_filter), exist_ok=True)

        spark = SparkSession(sc)

        train_df = spark.read.parquet(self.input()[0][0].path)
        account_df = spark.read.csv(self.input()[1][0].path, header=True, inferSchema=True)
        merchant_df = spark.read.csv(self.input()[1][1].path, header=True, inferSchema=True)

        train_df = train_df.join(account_df, train_df.account_id == account_df.account_id)
        train_df = train_df.join(merchant_df, train_df.merchant_id == merchant_df.merchant_id)

        train_df = train_df.select("account_idx", "merchant_idx", "visits", "buys")

        train_df.toPandas().to_csv(self.output().path, index=False)


class PrepareIfoodBinaryBuysInteractionsDataFrames(BasePrepareDataFrames):
    def requires(self):
        return GenerateIndicesForAccountsAndMerchantsOfInteractionsDataset(window_filter=self.window_filter),\
               IndexAccountsAndMerchantsOfInteractionsDataset(window_filter=self.window_filter)

    @property
    def dataset_dir(self) -> str:
        return os.path.join(DATASET_DIR, self.window_filter)

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


class PrepareIfoodAccountMatrixWithBinaryBuysDataFrames(BasePrepareDataFrames):
    split_per_user: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return GenerateIndicesForAccountsAndMerchantsOfInteractionsDataset(window_filter=self.window_filter), \
               IndexAccountsAndMerchantsOfInteractionsDataset(window_filter=self.window_filter)

    @property
    def dataset_dir(self) -> str:
        return os.path.join(DATASET_DIR, self.window_filter)

    @property
    def stratification_property(self) -> str:
        return "buys" if not self.split_per_user else None

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

    def _transform_data_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["buys"] > 0]
        df = df[["account_idx", "merchant_idx", "buys"]]
        df = df.groupby('account_idx')[['merchant_idx', 'buys']].apply(lambda x: x.values.tolist()).reset_index()
        df.columns = ["account_idx", "buys_per_merchant"]

        df["n_users"] = self.num_users
        df["n_items"] = self.num_businesses

        return df

    def read_data_frame(self) -> pd.DataFrame:
        df = pd.read_csv(self.input()[1].path)
        df["buys"] = (df["buys"] > 0).astype(float)
        if self.split_per_user:
            df = self._transform_data_frame(df)

        return df

    def transform_data_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.split_per_user:
            return self._transform_data_frame(df)
        return df



class PrepareIfoodIndexedOrdersTestData(BasePySparkTask):
    def requires(self):
        return FilterDataset(window_filter=self.window_filter), \
                GenerateIndicesForAccountsAndMerchantsOfInteractionsDataset(window_filter=self.window_filter)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, self.window_filter, "indexed_orders_test_data.parquet"))

    def main(self, sc: SparkContext, *args):
        os.makedirs(os.path.join(DATASET_DIR, self.window_filter), exist_ok=True)

        spark = SparkSession(sc)

        orders_df = spark.read.parquet(self.input()[0][1].path)
        account_df = spark.read.csv(self.input()[1][0].path, header=True, inferSchema=True)
        merchant_df = spark.read.csv(self.input()[1][1].path, header=True, inferSchema=True)

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


class ListAccountMerchantTuplesForIfoodIndexedOrdersTestData(BasePySparkTask):
    def requires(self):
        return PrepareIfoodIndexedOrdersTestData(self.window_filter)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, self.window_filter, "account_merchant_tuples_from_test_data.parquet"))

    def main(self, sc: SparkContext, *args):
        os.makedirs(os.path.join(DATASET_DIR, self.window_filter), exist_ok=True)

        spark = SparkSession(sc)

        df = spark.read.parquet(self.input().path)

        tuples_df = df.select("account_idx", "merchant_idx")

        tuples_df = tuples_df.union(df.select("account_idx", "merchant_idx_list")
                                    .withColumn("merchant_idx", explode(df.merchant_idx_list))
                                    .select("account_idx", "merchant_idx")).dropDuplicates()

        tuples_df.write.parquet(self.output().path)
