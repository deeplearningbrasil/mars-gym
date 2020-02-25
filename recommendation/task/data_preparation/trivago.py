import os
import re
from collections import Counter

import luigi
import math
import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import collect_set, collect_list, lit, sum, udf, concat_ws, col, count, abs, date_format, max, \
    from_utc_timestamp, expr
from pyspark.sql.functions import explode, posexplode
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.window import Window
from torchnlp.encoders import LabelEncoder
from torchnlp.encoders.text.static_tokenizer_encoder import StaticTokenizerEncoder
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pyspark.sql.types import ArrayType

from recommendation.task.data_preparation.base import BasePySparkTask, BasePrepareDataFrames
from recommendation.utils import parallel_literal_eval, datetime_to_shift, date_to_day_of_week, date_to_day_of_month, clean_filename, literal_eval_if_str
from pyspark.sql.window import Window
import pyspark.sql.functions as func

BASE_DIR: str = os.path.join("output", "trivago")
DATASET_DIR: str = os.path.join("output", "trivago", "dataset")
FILES_DIR: str = os.path.join("files")


class CheckDataset(luigi.Task):
    def output(self):
        return luigi.LocalTarget(os.path.join(BASE_DIR, "trivagoRecSysChallengeData2019_v2", "train.csv")), \
               luigi.LocalTarget(os.path.join(BASE_DIR, "trivagoRecSysChallengeData2019_v2", "item_metadata.csv"))

    def run(self):
        raise AssertionError(
            f"As seguintes pastas são esperadas com o dataset: {[output.path for output in self.output()]}")

class FilterDataset(BasePySparkTask):
    sample_size: int = luigi.IntParameter(default=0)
    filter_city: str = luigi.Parameter(default='all')

    def requires(self):
      return CheckDataset()

    def output(self):
      return luigi.LocalTarget(os.path.join(DATASET_DIR, clean_filename(self.filter_city), "train__size=%d__.csv" % (self.sample_size))),\
              luigi.LocalTarget(os.path.join(DATASET_DIR, clean_filename(self.filter_city), "item_metadata__size=%d__.csv" % (self.sample_size)))

    def main(self, sc: SparkContext, *args):
      os.makedirs(os.path.join(DATASET_DIR, clean_filename(self.filter_city)), exist_ok=True)

      spark = SparkSession(sc)
      train_df = spark.read.csv(self.input()[0].path, header=True, inferSchema=True)
      meta_df  = spark.read.csv(self.input()[1].path, header=True, inferSchema=True)

      # Filter dataset
      if self.filter_city != 'all':
        train_df     = train_df.filter(train_df.city==self.filter_city)
        reference_df = train_df.select("reference").distinct()
        meta_df      = meta_df.join(reference_df, meta_df.item_id == reference_df.reference).select("item_id","properties")

      if self.sample_size > 0:
        train_df = train_df.sort("timestamp", ascending=False).limit(self.sample_size)

      # Save
      train_df.toPandas().to_csv(self.output()[0].path, index=False)
      meta_df.toPandas().to_csv(self.output()[1].path, index=False)

class TransformMetaDataset(luigi.Task):
    sample_size: int = luigi.IntParameter(default=0)
    filter_city: str = luigi.Parameter(default='all')

    def requires(self):
      return FilterDataset(sample_size=self.sample_size, filter_city=self.filter_city)

    def output(self):
      return luigi.LocalTarget(os.path.join(DATASET_DIR, clean_filename(self.filter_city), "item_metadata_transform__size=%d__.csv" % (self.sample_size)))

    # 'a|b|c' -> ['a', 'b', 'c']
    #
    # 
    def split_df_columns(self, df, column):
      tf = CountVectorizer(tokenizer=lambda x: x.split("|"))
      tf_df = tf.fit_transform(df[column]).todense()
      tf_df = pd.DataFrame(tf_df, columns = sorted(tf.vocabulary_.keys()))
      return tf_df
      
    def run(self):
      df_meta = pd.read_csv(self.input()[1].path)  

      tf_prop_meta = self.split_df_columns(df_meta, 'properties')
      df_meta      = df_meta.join(tf_prop_meta).drop(['properties'], axis=1)
      
      
      df_meta.to_csv(self.output().path, index=False)

class TransformSessionDataset(luigi.Task):
    sample_size: int = luigi.IntParameter(default=0)
    filter_city: str = luigi.Parameter(default='all')

    def requires(self):
      return FilterDataset(sample_size=self.sample_size, filter_city=self.filter_city)

    def output(self):
      return luigi.LocalTarget(os.path.join(DATASET_DIR, clean_filename(self.filter_city), "train_transform__size=%d__.csv" % (self.sample_size))),\
              luigi.LocalTarget(os.path.join(DATASET_DIR, clean_filename(self.filter_city), "text_vocabulary__size=%d__.csv" % (self.sample_size)))

    def run(self):
      df        = pd.read_csv(self.input()[0].path)  

      # Transform impressions, prices
      df['impressions'] = df['impressions'].fillna("").apply(lambda x: [] if x == "" else [int(i) for i in x.split("|")] )
      df['prices']      = df['prices'].fillna("").apply(lambda x: [] if x == "" else [float(p) for p in x.split("|")])

      # Transform reference in action_type

      for ref in ["interaction item image", "search for poi", "interaction item rating", "clickout item", 
                  "interaction item deals", "change of sort order", "search for item", "search for destination", 
                    "filter selection", "interaction item info"]:
        # clickout item: user makes a click-out on the item and gets forwarded to a partner website. The reference value for this action is the item_id. Other items that were displayed to the user and their associated prices are listed under the ‘impressions’ and ‘prices’ column for this action.
        # interaction item rating: user interacts with a rating or review of an item.
        #   The reference value for this action is the item id.
        # interaction item info: user interacts with item information.
        #   The reference value for this action is the item id.
        # interaction item image: user interacts with an image of an item.
        #   The reference value for this action is the item id.
        # interaction item deals: user clicks on the view more deals button.
        #   The reference value for this action is the item id.
        # change of sort order: user changes the sort order.
        #   The reference value for this action is the sort order description.
        # filter selection: user selects a filter.
        #   The reference value for this action is the filter description.
        # search for item: user searches for an accommodation.
        #   The reference value for this action is the item id.
        # search for destination: user searches for a destination.
        #   The reference value for this action is the name of the destination.
        # search for poi: user searches for a point of interest (POI).
        #   The reference value for this action is the name of the POI.     
        df['reference_'+clean_filename(ref)] = df.apply(lambda row: row['reference'] if row['action_type'] == ref else np.nan, axis=1)


      # Transform columns with text
      columns_with_string = ["reference_search_for_poi","reference_change_of_sort_order",
                            "reference_search_for_destination","reference_filter_selection","current_filters"]
      # vocabulario
      vocab = []
      for c in columns_with_string:
        vocab  += df[c].fillna("").tolist()

      # Tokenizer
      tokenizer = StaticTokenizerEncoder(vocab, tokenize=lambda x: x.split("|"), min_occurrences=2, reserved_tokens=[])

      #Apply tokenizer
      for text_column in columns_with_string:
        df[text_column] = tokenizer.batch_encode(df[text_column].fillna(""))[0].cpu().detach().numpy().tolist()
        df[text_column + '_max_words'] = len(df[text_column][0])

      # Save
      df.to_csv(self.output()[0].path, index=False)
      pd.DataFrame(tokenizer.vocab, columns=['vocabulary']).to_csv(self.output()[1].path)
      
class GenerateIndicesDataset(BasePySparkTask):
    sample_size: int = luigi.IntParameter(default=0)
    filter_city: str = luigi.Parameter(default='all')

    def requires(self):
      return FilterDataset(sample_size=self.sample_size, filter_city=self.filter_city)

    def output(self):
      return luigi.LocalTarget(os.path.join(DATASET_DIR, clean_filename(self.filter_city), "item_indices__size=%d__.csv" % (self.sample_size))),\
            luigi.LocalTarget(os.path.join(DATASET_DIR, clean_filename(self.filter_city),  "user_indices__size=%d__.csv" % (self.sample_size))),\
            luigi.LocalTarget(os.path.join(DATASET_DIR, clean_filename(self.filter_city),  "session_indices__size=%d__.csv" % (self.sample_size))),\
            luigi.LocalTarget(os.path.join(DATASET_DIR, clean_filename(self.filter_city),  "action_type_indices__size=%d__.csv" % (self.sample_size))),\
            luigi.LocalTarget(os.path.join(DATASET_DIR, clean_filename(self.filter_city),  "platform_indices__size=%d__.csv" % (self.sample_size))),\
            luigi.LocalTarget(os.path.join(DATASET_DIR, clean_filename(self.filter_city),  "city_indices__size=%d__csv" % (self.sample_size))),\
            luigi.LocalTarget(os.path.join(DATASET_DIR, clean_filename(self.filter_city),  "device_indices__size=%d__.csv" % (self.sample_size))),\

    def main(self, sc: SparkContext, *args):
      os.makedirs(DATASET_DIR, exist_ok=True)

      spark = SparkSession(sc)

      # Load
      train_df = spark.read.csv(self.input()[0].path, header=True, inferSchema=True)
      meta_df  = spark.read.csv(self.input()[1].path, header=True, inferSchema=True)

      # Extract
      item_idx_df    = meta_df.select("item_id").distinct().toPandas()
      user_idx_df    = train_df.select("user_id").distinct().toPandas()
      session_idx_df = train_df.select("session_id").distinct().toPandas()
      action_type_idx_df = train_df.select("action_type").distinct().toPandas()
      platform_idx_df = train_df.select("platform").distinct().toPandas()
      city_idx_df     = train_df.select("city").distinct().toPandas()
      device_idx_df   = train_df.select("device").distinct().toPandas()

      # Save
      item_idx_df.shift(periods=1, fill_value=0).to_csv(self.output()[0].path, index_label="item_idx")
      user_idx_df.to_csv(self.output()[1].path, index_label="user_idx")
      session_idx_df.to_csv(self.output()[2].path, index_label="session_idx")
      action_type_idx_df.to_csv(self.output()[3].path, index_label="action_type_idx")
      platform_idx_df.to_csv(self.output()[4].path, index_label="platform_idx")
      city_idx_df.to_csv(self.output()[5].path, index_label="city_idx")
      device_idx_df.to_csv(self.output()[6].path, index_label="device_idx")

class CreateIndexDataset(BasePySparkTask):
    sample_size: int = luigi.IntParameter(default=0)
    filter_city: str = luigi.Parameter(default='all')

    def requires(self):
      return  TransformSessionDataset(sample_size=self.sample_size, filter_city=self.filter_city),\
              TransformMetaDataset(sample_size=self.sample_size, filter_city=self.filter_city),\
              GenerateIndicesDataset(sample_size=self.sample_size, filter_city=self.filter_city)

    def output(self):
      return luigi.LocalTarget(os.path.join(DATASET_DIR, clean_filename(self.filter_city), "train_indexed__size=%d__.csv" % (self.sample_size))),\
              luigi.LocalTarget(os.path.join(DATASET_DIR, clean_filename(self.filter_city), "item_metadata_indexed__size=%d__.csv" % (self.sample_size)))


    def main(self, sc: SparkContext, *args):
      os.makedirs(DATASET_DIR, exist_ok=True)
      spark = SparkSession(sc)

      train_df           = spark.read.csv(self.input()[0][0].path, header=True, inferSchema=True)
      meta_df            = spark.read.csv(self.input()[1].path, header=True, inferSchema=True)

      item_idx_df        = spark.read.csv(self.input()[2][0].path, header=True, inferSchema=True)
      user_idx_df        = spark.read.csv(self.input()[2][1].path, header=True, inferSchema=True)
      session_idx_df     = spark.read.csv(self.input()[2][2].path, header=True, inferSchema=True)
      action_type_idx_df = spark.read.csv(self.input()[2][3].path, header=True, inferSchema=True)
      platform_idx_df    = spark.read.csv(self.input()[2][4].path, header=True, inferSchema=True)
      city_idx_df        = spark.read.csv(self.input()[2][5].path, header=True, inferSchema=True)
      device_idx_df      = spark.read.csv(self.input()[2][6].path, header=True, inferSchema=True)

      
      # Join with item_idx 
      for c in ["reference_interaction_item_image","reference_interaction_item_rating","reference_clickout_item",
                "reference_interaction_item_deals","reference_search_for_item","reference_interaction_item_info"]:
        train_df = train_df.\
                   join(item_idx_df.withColumnRenamed("item_id",c).withColumnRenamed("item_idx",c+"_idx"), c,how='left').\
                   fillna(0)

      # Joint others index
      train_df = train_df.\
        join(user_idx_df, "user_id").\
        join(session_idx_df, "session_id").\
        join(action_type_idx_df, "action_type").\
        join(platform_idx_df, "platform").\
        join(city_idx_df, "city").\
        join(device_idx_df, "device")

      # Index impressions
      #item_idx_df = item_idx_df.set_index('item_id')


      # Joint meta
      meta_df = meta_df.join(item_idx_df, "item_id")


      # Save
      df = train_df.select("timestamp","step", "user_idx", "session_idx", "action_type_idx", 
                      "reference_search_for_poi","reference_change_of_sort_order",
                      "reference_search_for_destination","reference_filter_selection",
                      "reference_interaction_item_image_idx","reference_interaction_item_rating_idx","reference_clickout_item_idx",
                      "reference_interaction_item_deals_idx","reference_search_for_item_idx","reference_interaction_item_info_idx",
                      "platform_idx", "city_idx", "device_idx", "current_filters", "impressions", "prices")\
              .toPandas()
      
      df.to_csv(self.output()[0].path, index=False)
      meta_df.toPandas().to_csv(self.output()[1].path, index=False)

class CreateAggregateIndexDataset(BasePySparkTask):
    sample_size: int = luigi.IntParameter(default=0)
    filter_city: str = luigi.Parameter(default='all')

    def requires(self):
      return  CreateIndexDataset(sample_size=self.sample_size, filter_city=self.filter_city)

    def output(self):
      return luigi.LocalTarget(os.path.join(DATASET_DIR, clean_filename(self.filter_city), "train__agg_indexed__size=%d__.csv" % (self.sample_size)))

    def main(self, sc: SparkContext, *args):
      os.makedirs(DATASET_DIR, exist_ok=True)
      spark = SparkSession(sc)

      # ['timestamp', 'step', 'user_idx', 'session_idx', 'action_type_idx',
      #  'reference_search_for_poi', 'reference_change_of_sort_order',
      #  'reference_search_for_destination', 'reference_filter_selection',
      #  'reference_interaction_item_image_idx',
      #  'reference_interaction_item_rating_idx', 'reference_clickout_item_idx',
      #  'reference_interaction_item_deals_idx', 'reference_search_for_item_idx',
      #  'reference_interaction_item_info_idx', 'platform_idx', 'city_idx',
      #  'device_idx', 'current_filters', 'impressions', 'prices'],
      train_df = spark.read.csv(self.input()[0].path, header=True, inferSchema=True)

      win_over_session = Window.partitionBy('session_idx').orderBy('timestamp')\
                          .rangeBetween(Window.unboundedPreceding, -1)
      win_over_session_with_current = Window.partitionBy('session_idx').orderBy('timestamp')\
                          .rangeBetween(Window.unboundedPreceding, 0)

      def to_array(xs):
          return [literal_eval_if_str(c) if isinstance(literal_eval_if_str(c), int) else literal_eval_if_str(c)[0] for c in literal_eval_if_str(xs)] if xs is not None else None
      to_array_udf = udf(lambda y: to_array(y), ArrayType(IntegerType()))


      df_group = train_df.\
                withColumn("timestamp_diff", F.sum(col("timestamp")).over(win_over_session_with_current)-col("timestamp")).\
                withColumn("step", max("step").over(win_over_session)+lit(1)).\
                withColumn("list_reference_search_for_poi", to_array_udf(collect_list("reference_search_for_poi").over(win_over_session))).\
                withColumn("list_reference_change_of_sort_order", to_array_udf(collect_list("reference_change_of_sort_order").over(win_over_session))).\
                withColumn("list_reference_search_for_destination", to_array_udf(collect_list("reference_search_for_destination").over(win_over_session))).\
                withColumn("list_reference_filter_selection", to_array_udf(collect_list("reference_filter_selection").over(win_over_session))).\
                withColumn("list_reference_interaction_item_image_idx", to_array_udf(collect_list("reference_interaction_item_image_idx").over(win_over_session))).\
                withColumn("list_reference_interaction_item_rating_idx", to_array_udf(collect_list("reference_interaction_item_rating_idx").over(win_over_session))).\
                withColumn("list_reference_clickout_item_idx", to_array_udf(collect_list("reference_clickout_item_idx").over(win_over_session))).\
                withColumn("list_reference_interaction_item_deals_idx", to_array_udf(collect_list("reference_interaction_item_deals_idx").over(win_over_session))).\
                withColumn("list_reference_search_for_item_idx", to_array_udf(collect_list("reference_search_for_item_idx").over(win_over_session))).\
                withColumn("list_reference_interaction_item_info_idx", to_array_udf(collect_list("reference_interaction_item_info_idx").over(win_over_session))).\
                withColumn("list_action_type_idx", collect_list("action_type_idx").over(win_over_session)).\
                withColumn("list_current_filters", to_array_udf(collect_list("current_filters").over(win_over_session))).\
                orderBy("session_idx", "timestamp")

      # Filter only action type - clickout item
      df_group = df_group.filter(df_group.action_type_idx == 3) # 3,clickout item

      df_group = df_group.select('timestamp', 'timestamp_diff', 'step', 'user_idx', 'session_idx', 'action_type_idx',
                                'list_action_type_idx', 'list_reference_search_for_poi', 'list_reference_change_of_sort_order',
                                'list_reference_search_for_destination', 'list_reference_filter_selection', 'list_reference_interaction_item_image_idx',
                                'list_reference_interaction_item_rating_idx', 'list_reference_clickout_item_idx', 'list_reference_interaction_item_deals_idx',
                                'list_reference_search_for_item_idx', 'list_reference_interaction_item_info_idx', 'list_action_type_idx', 
                                'list_current_filters', 'device_idx', 'current_filters', 'impressions', 'prices')
      df_group.toPandas().to_csv(self.output().path, index=False)
