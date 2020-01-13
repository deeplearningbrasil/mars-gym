import math
import os
import re
from collections import Counter
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

import luigi
import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set, collect_list, lit, sum, udf, concat_ws, col, count, abs
from pyspark.sql.functions import explode, posexplode
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql import functions as F
from torchnlp.encoders import LabelEncoder
from torchnlp.encoders.text.static_tokenizer_encoder import StaticTokenizerEncoder
from unidecode import unidecode
from pyspark.sql.functions import when   

from recommendation.task.data_preparation.base import BasePySparkTask, BasePrepareDataFrames, BaseDownloadDataset
from recommendation.utils import parallel_literal_eval
from collections import Counter
import requests

BASE_DIR: str = os.path.join("output", "ifood")
DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")
EMBEDDING_DIR: str = os.path.join("output", "embeddings")

class CheckDataset(luigi.Task):
    def output(self):
        return luigi.LocalTarget(os.path.join(BASE_DIR, "ufg_dataset_all", "info_availability")), \
               luigi.LocalTarget(os.path.join(BASE_DIR, "ufg_dataset_all", "info_delivery_time")), \
               luigi.LocalTarget(os.path.join(BASE_DIR, "ufg_dataset_all", "info_items")), \
               luigi.LocalTarget(os.path.join(BASE_DIR, "ufg_dataset_all", "info_menu")), \
               luigi.LocalTarget(os.path.join(BASE_DIR, "ufg_dataset_all", "info_restaurant")), \
               luigi.LocalTarget(os.path.join(BASE_DIR, "ufg_dataset_all", "info_review")), \
               luigi.LocalTarget(os.path.join(BASE_DIR, "ufg_dataset_all", "info_session"))

    def run(self):
        raise AssertionError(
            f"As seguintes pastas são esperadas com o dataset: {[output.path for output in self.output()]}")


def date_to_day_of_week(date: str) -> int:
    return int(datetime.strptime(date, '%Y-%m-%d').strftime('%w'))


def datetime_to_shift(datetime_: str) -> str:
    datetime_: datetime = datetime_ if isinstance(datetime_, datetime) else datetime.strptime(datetime_,
                                                                                              '%Y-%m-%d %H:%M:%S')
    day_of_week = int(datetime_.strftime('%w'))
    # 0 - 4:59h - dawn
    # 5 - 9:59h - breakfast
    # 10 - 13:59h - lunch
    # 14 - 16:59h - snack
    # 17 - 23:59h - dinner
    if 0 <= datetime_.hour <= 4:
        shift = "dawn"
    elif 5 <= datetime_.hour <= 9:
        shift = "breakfast"
    elif 10 <= datetime_.hour <= 13:
        shift = "lunch"
    elif 14 <= datetime_.hour <= 16:
        shift = "snack"
    else:
        shift = "dinner"
    return "%s %s" % ("weekday" if day_of_week < 4 else "weekend", shift)


class CreateShiftIndices(BasePySparkTask):
    def requires(self):
        return CheckDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "shifts.csv"))

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        spark = SparkSession(sc)

        availability_df = spark.read.parquet(self.input()[0].path)

        shift_df = availability_df.select("shift").distinct().toPandas()

        shift_df.to_csv(self.output().path, index_label="shift_idx")


class AddShiftIdxAndFixWeekDayForAvailabilityDataset(BasePySparkTask):
    def requires(self):
        return CheckDataset(), CreateShiftIndices()

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "availability"))

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        spark = SparkSession(sc)

        df = spark.read.parquet(self.input()[0][0].path)

        df = df.withColumn("day_of_week", df["day_of_week"] - 1)

        shift_df = spark.read.csv(self.input()[1].path, header=True, inferSchema=True)
        df = df.join(shift_df, "shift")

        df.write.parquet(self.output().path)


class AddShiftAndWeekDayToSessionDataset(BasePySparkTask):
    def requires(self):
        return CheckDataset(), CreateShiftIndices()

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "session"))

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        spark = SparkSession(sc)

        date_to_day_of_week_udf = udf(date_to_day_of_week, IntegerType())
        datetime_to_shift_udf = udf(datetime_to_shift, StringType())

        df = spark.read.parquet(self.input()[0][6].path)

        df = df.withColumn("day_of_week", date_to_day_of_week_udf(df.dt_partition))
        df = df.withColumn("shift", datetime_to_shift_udf(df.click_timestamp))

        shift_df = spark.read.csv(self.input()[1].path, header=True, inferSchema=True)
        df = df.join(shift_df, "shift")

        df.write.parquet(self.output().path)


class PrepareRestaurantContentDataset(BasePySparkTask):
    def requires(self):
        return CheckDataset(), AddShiftIdxAndFixWeekDayForAvailabilityDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "restaurants_with_contents.csv"))

    def main(self, sc: SparkContext, *args):
        spark = SparkSession(sc)

        restaurant_df = spark.read.parquet(self.input()[0][4].path)
        availability_df = spark.read.parquet(self.input()[1].path)
        menu_df = spark.read.parquet(self.input()[0][3].path)
        item_df = spark.read.parquet(self.input()[0][2].path)

        availability_df = availability_df.groupBy("merchant_id") \
            .agg(collect_set("day_of_week").alias("days_of_week"), collect_set("shift").alias("shifts"),
                 collect_set("shift_idx").alias("shift_idx_list"))

        menu_df = menu_df.groupBy("merchant_id").agg(collect_set("category_name").alias("category_names"))
        menu_df = menu_df.withColumn("category_names", concat_ws("|", "category_names"))

        item_df = item_df.withColumn("item_text", concat_ws("\n", item_df.item_name, item_df.item_description))
        item_df = item_df.groupBy("merchant_id").agg(collect_list("item_text").alias("item_texts"))
        item_df = item_df.withColumn("item_texts", concat_ws("\n\n", "item_texts"))

        menu_df = menu_df.join(item_df, on="merchant_id", how="full_outer")
        menu_df = menu_df.withColumn("menu_full_text", concat_ws("\n\n", "item_texts"))

        restaurant_df = restaurant_df.join(availability_df, on="merchant_id", how="left")
        restaurant_df = restaurant_df.join(menu_df.select("merchant_id", "category_names", "menu_full_text"),
                                           on="merchant_id", how="left")

        restaurant_df.toPandas().to_csv(self.output().path, index=False)



class DownloadStopWords(luigi.Task):
    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "stopwords_pt.txt"))

    def run(self):
        os.makedirs(DATASET_DIR, exist_ok=True)

        url  = "https://raw.githubusercontent.com/stopwords-iso/stopwords-pt/master/stopwords-pt.txt"
        file = requests.get(url)

        with open(self.output().path, 'wb') as output:
            output.write(file.content)

        print(self.output().path)

class ProcessRestaurantContentDataset(luigi.Task):
    menu_text_length: int = luigi.IntParameter(default=5000)
    description_text_length: int = luigi.IntParameter(default=200)
    category_text_length: int = luigi.IntParameter(default=250)

    def requires(self):
        return PrepareRestaurantContentDataset(), DownloadStopWords()
    
    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "restaurants_with_processed_contents.csv")), \
               luigi.LocalTarget(os.path.join(DATASET_DIR, "restaurant_text_vocabulary.csv"))

    def tokenizer(self, text):
        # print(text)

        text = str(text)

        # # Remove acentuação
        text = unidecode(text)

        # # lowercase
        text = text.lower()

        # #remove tags
        text = re.sub("<!--?.*?-->", "", text)

        # # remove special characters and digits
        text = re.sub("(\\d|\\W)+", " ", text)
        text = re.sub('[^A-Za-z0-9]+', ' ', text)

        # # punk
        text = re.sub(r'[?|!|\'|#]', r'', text)
        text = re.sub(r'[.|,|:|)|(|\|/]', r' ', text)

        # Clean onde
        tokens = [t.strip() for t in text.split() if len(t) > 1]
        
        # remove stopwords
        stopwords = self.load_stopwords()
        tokens    = [t for t in tokens if t not in stopwords]

        if len(tokens) == 0:
            tokens.append("<pad>")
        # print(tokens)
        # print("")
        # if len(tokens) < 2:
        #    print(tokens)
        return tokens

    def load_stopwords(self):
        if not hasattr(self, "_stopwords"):
            with open(self.input()[1].path, "r") as file_handler:
                self._stopwords = [unidecode(line.strip()) for line in file_handler.readlines()]

        return self._stopwords

    def run(self):
        #spark = SparkSession(sc)
        context = pd.read_csv(self.input()[0].path)

        del context['item_imagesurl']
        context['menu_full_text']   = context['menu_full_text'].str[:self.menu_text_length].replace(',', ' ')
        context['description']      = context['description'].str[:self.description_text_length].replace(',', ' ')
        context['category_names']   = context['category_names'].str[:self.category_text_length].replace(',', ' ')
        
        for column in ['days_of_week', 'shifts']:
            context[column] = parallel_literal_eval(context[column].fillna('[]'))
            elements        = np.unique(context[column].sum())

            for el in elements:
               context[el] = context[column].apply(lambda x: ((el in x) * 1.0))

        context['avg_score'] = context['avg_score'].fillna(-1)
        context['category_names'] = context['category_names'].fillna('NAN').str.replace('|', ' ')

        encoder = LabelEncoder(context['dish_description'].values)
        context['dish_description'] = encoder.batch_encode(context['dish_description'])

        vocab = context['description'].values.tolist() + context['trading_name'].values.tolist() + \
                context['category_names'].values.tolist() + context['menu_full_text'].values.tolist()

        context = context.fillna('NAN').replace(r'^\s*$', 'NAN', regex=True)
        tokenizer = StaticTokenizerEncoder(
            vocab, tokenize=self.tokenizer, min_occurrences=2)  # int(len(vocab)*0.01)

        for text_column in ['trading_name', 'description', 'category_names', "menu_full_text"]:
            context[text_column] = tokenizer.batch_encode(context[text_column])[0].cpu().detach().numpy().tolist()
            context[text_column + '_max_words'] = len(context[text_column][0])
        
        restaurant_features = ['dish_description', 'price_range', 'avg_score', \
                                'latitude', 'longitude', 0, 1, 2, 3, 4, 5, 6, \
                                'weekday breakfast', 'weekday dawn', 'weekday dinner', \
                                'weekday lunch', 'weekday snack', 'weekend breakfast', \
                                'weekend dawn', 'weekend dinner', 'weekend lunch', \
                                'weekend snack']
        
        context['restaurant_complete_info'] = context[restaurant_features].values.tolist()

        context['vocab_size'] = len(tokenizer.vocab)
        context['non_textual_input_dim'] = len(restaurant_features)

        context.to_csv(self.output()[0].path, index=False)
        pd.DataFrame(tokenizer.vocab, columns=['vocabulary']).to_csv(self.output()[1].path)



class BuildEmbeddingVocabulary(luigi.Task):
    menu_text_length: int = luigi.IntParameter(default=5000)
    description_text_length: int = luigi.IntParameter(default=200)
    category_text_length: int = luigi.IntParameter(default=250)
    filename: str = luigi.ChoiceParameter(choices=["cc.pt.300.vec"], default="cc.pt.300.vec")

    def requires(self):
        return ProcessRestaurantContentDataset(menu_text_length=self.menu_text_length, 
                                                description_text_length=self.description_text_length, 
                                                category_text_length=self.category_text_length)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "restaurant_text_vocabulary_vec.npy"))

    def load_embvec(self):
        word2vec    = {}
        words_count = 0

        with open(f'{EMBEDDING_DIR}/{self.filename}', 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                vect = np.array(line[1:]).astype(np.float)

                word2vec[word] = vect
                words_count += 1

                #if words_count > 100:
                #    break

        return word2vec

    def run(self):
        df_vocab        = pd.read_csv(self.input()[1].path)
        word2vec        = self.load_embvec()
        emb_dim         = 300

        matrix_len      = len(df_vocab)
        weights_matrix  = np.zeros((matrix_len, emb_dim))
        words_found     = 0

        for i, row in df_vocab.iterrows():
            try: 
                weights_matrix[i] = word2vec[row.vocabulary]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
                        
        print("Words found in Embedding: ", (words_found/matrix_len)*100)

        np.save(self.output().path, weights_matrix)

class SplitSessionDataset(BasePySparkTask):
    test_size: float = luigi.FloatParameter(default=0.10)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    def requires(self):
        return AddShiftAndWeekDayToSessionDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "session_train_%.2f_k=%d_s=%d" % (self.test_size,
                                                                                             self.minimum_interactions,
                                                                                             self.sample_size))), \
               luigi.LocalTarget(os.path.join(DATASET_DIR, "session_test_%.2f_s=%d" % (self.test_size, self.sample_size)))

    def filter_train_session(self, df: pd.DataFrame, minimum_interactions: int) -> pd.DataFrame:
        df_account_buy    = df.groupBy("account_id").agg(sum(df.buy).alias("count_buy")).cache()
        df_account_buy    = df_account_buy.filter(col('count_buy') >= minimum_interactions)

        df_merchant_visit = df.groupBy("merchant_id").count()
        df_merchant_visit = df_merchant_visit.filter(col('count') >= minimum_interactions)

        df = df \
            .join(df_account_buy, "account_id", how="inner") \
            .join(df_merchant_visit, "merchant_id", how="inner")

        return df

    def main(self, sc: SparkContext, *args):
        spark = SparkSession(sc)

        df = spark.read.parquet(self.input().path)
        df = df.filter(df.account_id.isNotNull()).dropDuplicates()

        if self.sample_size > 0:
            df = df.sort("click_timestamp").limit(self.sample_size)

        count  = df.count()
        n_test = math.ceil(self.test_size * count)

        train_df = df.sort("click_timestamp").limit(count - n_test).cache()
        test_df  = df.sort("click_timestamp", ascending=False).limit(n_test).cache()

        train_df = self.filter_train_session(train_df, self.minimum_interactions)

        train_df.write.parquet(self.output()[0].path)
        test_df.write.parquet(self.output()[1].path)


class AddVisitsBuysForInteractionDataset(luigi.Task):
    test_size: float = luigi.FloatParameter(default=0.10)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    def requires(self):
        return SplitSessionDataset(test_size=self.test_size, 
                                    sample_size=self.sample_size, 
                                    minimum_interactions=self.minimum_interactions)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "session_train_buys_visits_%.2f_k=%d_s=%d" % (self.test_size,
                                                                                        self.minimum_interactions, self.sample_size))), \
                luigi.LocalTarget(os.path.join(DATASET_DIR, "session_test_buys_visits_%.2f_k=%d_s=%d" % (self.test_size,
                                                                                        self.minimum_interactions, self.sample_size)))

    def add_visits_buys(self, df):
        user_total_buys     = []
        user_total_visits   = []
        item_total_buys     = []
        item_total_visits   = []        
        hist_buys           = []
        hist_visits         = []

        user_buys_dict      = defaultdict(int)
        user_visits_dict    = defaultdict(int)

        item_buys_dict      = defaultdict(int)
        item_visits_dict    = defaultdict(int)

        hist_buys_dict      = defaultdict(int)
        hist_visits_dict    = defaultdict(int)

        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            merchant, account = row['merchant_id'], row['account_id']
            hist_tup = tuple([merchant, account])
            
            user_total_buys.append(user_buys_dict[account])
            user_total_visits.append(user_visits_dict[account])
            item_total_buys.append(item_buys_dict[merchant])
            item_total_visits.append(item_visits_dict[merchant])
            hist_buys.append(hist_buys_dict[hist_tup])
            hist_visits.append(hist_visits_dict[hist_tup])
            
            user_buys_dict[account]     += row['buy']
            user_visits_dict[account]   += 1
            item_buys_dict[merchant]    += row['buy']
            item_visits_dict[merchant]  += 1            
            hist_buys_dict[hist_tup]    += row['buy']
            hist_visits_dict[hist_tup]  += 1
            
        df['user_total_buys']   = user_total_buys
        df['user_total_visits'] = user_total_visits
        df['item_total_buys']   = item_total_buys
        df['item_total_visits'] = item_total_visits        
        df['hist_buys']         = hist_buys
        df['hist_visits']       = hist_visits    

        return df                                                                                

    def run(self):
        #spark = SparkSession(sc)

        train_df = pd.read_parquet(self.input()[0].path).sort_values("click_timestamp")
        test_df  = pd.read_parquet(self.input()[1].path).sort_values("click_timestamp")
        #df = df.filter(df.account_id.isNotNull()).dropDuplicates()

        train_df = self.add_visits_buys(train_df)
        test_df  = self.add_visits_buys(test_df)

        train_df.to_parquet(self.output()[0].path)
        test_df.to_parquet(self.output()[1].path)

class GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(BasePySparkTask):
    test_size: float = luigi.FloatParameter(default=0.10)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    def requires(self):
        return ProcessRestaurantContentDataset(), \
               SplitSessionDataset(test_size=self.test_size,
                                   minimum_interactions=self.minimum_interactions,
                                   sample_size=self.sample_size)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(DATASET_DIR, "accounts_for_session_train_%.2f_k=%d_s=%d.csv" % (self.test_size,
                                                                                    self.minimum_interactions,
                                                                                    self.sample_size))), \
               luigi.LocalTarget(
                   os.path.join(DATASET_DIR, "merchants_for_session_train_%.2f_k=%d_s=%d.csv" % (self.test_size,
                                                                                            self.minimum_interactions,
                                                                                            self.sample_size)))

    def main(self, sc: SparkContext, *args):
        spark = SparkSession(sc)

        train_df = spark.read.parquet(self.input()[1][0].path)
        restaurant_df = spark.read.csv(self.input()[0][0].path, header=True)

        account_df = train_df.select("account_id").distinct()
        merchant_df = train_df.select("merchant_id").distinct()

        merchant_df = merchant_df.join(restaurant_df, "merchant_id")

        account_df.toPandas().to_csv(self.output()[0].path, index_label="account_idx")
        merchant_df.toPandas().to_csv(self.output()[1].path, index_label="merchant_idx")


class IndexAccountsAndMerchantsOfSessionTrainDataset(BasePySparkTask):
    test_size: float = luigi.FloatParameter(default=0.10)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    def requires(self):
        return AddVisitsBuysForInteractionDataset(
                    test_size=self.test_size, 
                    minimum_interactions=self.minimum_interactions,
                    sample_size=self.sample_size), \
               GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(
                   test_size=self.test_size,
                   minimum_interactions=self.minimum_interactions,
                   sample_size=self.sample_size)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "indexed_session_train_%.2f_k=%d_s=%d" % (self.test_size,
                                                                                                     self.minimum_interactions,
                                                                                                     self.sample_size
                                                                                                    )))

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        spark = SparkSession(sc)

        train_df = spark.read.parquet(self.input()[0][0].path)
        account_df = spark.read.csv(self.input()[1][0].path, header=True, inferSchema=True)
        merchant_df = spark.read.csv(self.input()[1][1].path, header=True, inferSchema=True) \
            .select("merchant_idx", "merchant_id")

        train_df = train_df.join(account_df, "account_id")
        train_df = train_df.join(merchant_df, "merchant_id")

        train_df.write.parquet(self.output().path)


class LoggingPolicyPsDataset(BasePySparkTask):
    test_size: float = luigi.FloatParameter(default=0.10)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    def requires(self):
        return IndexAccountsAndMerchantsOfSessionTrainDataset(
                            test_size=self.test_size, 
                            minimum_interactions=self.minimum_interactions,
                            sample_size=self.sample_size)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "logging_policy_ps_session_train_%.2f_k=%d_s=%d.csv" % (self.test_size,
                                                                                                self.minimum_interactions, self.sample_size)))

    def _evaluate_logging_policy(self, orders_df):

        orders_df = orders_df.withColumn("visit", lit(1))

        # Group per account and merchant

        acc_mer_count = orders_df.groupBy("account_idx", "merchant_idx") \
                            .agg(sum("visit").alias("account_merchant_visits"))

        # Group per Account
        acc_count = orders_df.groupBy('account_idx') \
                        .agg(sum("visit").alias("account_visits"))

        p0 = acc_mer_count.join(acc_count, "account_idx", how="inner")
        p0 = p0.withColumn("ps", p0.account_merchant_visits/p0.account_visits)

        p0 = p0.select("account_idx", "merchant_idx", "ps").toPandas()

        return p0

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)

        spark    = SparkSession(sc)
        train_df = spark.read.parquet(self.input().path).orderBy(col('click_timestamp'))

        # Calculate P0 (logging policy)
        p0 = self._evaluate_logging_policy(train_df)

        p0.to_csv(self.output().path, index=False)


class CreateIntraSessionInteractionDataset(BasePySparkTask):
    test_size: float = luigi.FloatParameter(default=0.10)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    def requires(self):
        return IndexAccountsAndMerchantsOfSessionTrainDataset(
                    test_size=self.test_size, 
                    minimum_interactions=self.minimum_interactions,
                    sample_size=self.sample_size)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "indexed_intra_session_train_%.2f_k=%d_s=%d" % (self.test_size, self.minimum_interactions, self.sample_size)))


    def get_df_tuple_probs(self, df):

        df_tuple_count  = df.groupby("merchant_idx_A","merchant_idx_B").count()
        df_count        = df.groupby("merchant_idx_A").count()\
                            .withColumnRenamed("count", "total").withColumnRenamed("merchant_idx_A", "_merchant_idx_A")

        df_join         = df_tuple_count.join(df_count, df_tuple_count.merchant_idx_A == df_count._merchant_idx_A).cache()
        df_join         = df_join.withColumn("prob", col("count")/col("total"))

        df_join  = df_join.select("merchant_idx_A", 'merchant_idx_B', 'total', 'prob')\
                    .withColumnRenamed("merchant_idx_A", "_merchant_idx_A")\
                    .withColumnRenamed("merchant_idx_B", "_merchant_idx_B")\
                    .withColumnRenamed("total", "total_ocr").cache()
        
        return df_join

    def main(self, sc: SparkContext, *args):
        os.makedirs(DATASET_DIR, exist_ok=True)
        
        #parans
        total_itens_per_session  = 3
        total_itens_interactions = 30


        spark    = SparkSession(sc)

        train_df = spark.read.parquet(self.input().path).orderBy(col('click_timestamp'))

        df       = train_df.groupby("session_id").agg(
                    collect_list("merchant_idx").alias("merchant_idxs"), 
                    count("merchant_idx").alias("total"), 
                    sum("buy").alias("buy"))

        # Filter Interactions
        df = df.filter(df.total >= total_itens_per_session).filter(df.buy > 1).cache()

        # Filter position in list
        df_pos = df.select(col('session_id').alias('_session_id'), 
                                    posexplode(df.merchant_idxs))

        # Explode A
        df = df.withColumn("merchant_idx_A", explode(df.merchant_idxs))
        df = df.join(df_pos, 
                    (df.session_id == df_pos._session_id) & (df.merchant_idx_A == df_pos.col))\
                .select('session_id', 'merchant_idx_A', 'pos', 'merchant_idxs')\
                .withColumnRenamed('pos', 'pos_A')

        # Explode B
        df = df.withColumn("merchant_idx_B", explode(df.merchant_idxs))
        df = df.join(df_pos, 
                    (df.session_id == df_pos._session_id) & (df.merchant_idx_B == df_pos.col))\
                .withColumnRenamed('pos', 'pos_B')

        df = df.withColumn("relative_pos", abs(df.pos_A - df.pos_B))

        # Filter  distincts
        df = df.select('session_id', 'merchant_idx_A', 'pos_A', 'merchant_idx_B', 'pos_B', 'relative_pos')\
                .distinct()\
                .filter(df.merchant_idx_A != df.merchant_idx_B).cache()

        # Filter duplicates
        udf_join = F.udf(lambda s,x,y : "_".join(sorted([str(s), str(x),str(y)])) , StringType())
        df = df.withColumn('key', udf_join('session_id', 'merchant_idx_A','merchant_idx_B'))

        df = df.dropDuplicates(["key"])

        # Calculate and filter probs ocorrence
        df_probs = self.get_df_tuple_probs(df)

        df = df.join(df_probs, (df.merchant_idx_A == df_probs._merchant_idx_A) & (df.merchant_idx_B == df_probs._merchant_idx_B))
        df = df.filter(col("total_ocr") >= total_itens_interactions)

        # Save
        df = df.select('session_id',  'merchant_idx_A', 'pos_A', 
                        'merchant_idx_B', 'pos_B', 'relative_pos', 'prob')

        df.write.parquet(self.output().path)

class CreateInteractionDataset(BasePySparkTask):
    test_size: float = luigi.FloatParameter(default=0.10)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    def requires(self):
        return IndexAccountsAndMerchantsOfSessionTrainDataset(test_size=self.test_size,
                                                              minimum_interactions=self.minimum_interactions,
                                                              sample_size=self.sample_size)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "interactions_train_%.2f_k=%d_s=%d" % (
            self.test_size, self.minimum_interactions, self.sample_size)))

    def main(self, sc: SparkContext, *args):
        spark = SparkSession(sc)

        # Mode Value Spark
        def mode_value(ids):
            c = Counter()
            for cnid in ids:
                c[cnid] += 1
            return c.most_common(1)[0][0]

        mode_value_udf = udf(mode_value, IntegerType())
        
        train_df = spark.read.parquet(self.input().path)
        
        # Group per Account
        train_df_grouped_users = train_df.withColumn("user_total_buys", col("buy")) \
            .withColumn("user_total_visits", lit(1)) \
            .groupBy("account_id") \
            .agg(sum("user_total_buys").alias("user_total_buys"), 
                 sum("user_total_visits").alias("user_total_visits")) 
            
            
        # Group per Account and Merchant
        train_df_grouped = train_df.withColumn("visit", lit(1)) \
            .groupBy("account_id", "account_idx",
                     "merchant_id", "merchant_idx") \
            .agg(sum("visit").alias("visits"), 
                 sum("buy").alias("buys"),
                 mode_value_udf(collect_list("shift_idx")).alias("mode_shift_idx"),
                 mode_value_udf(collect_list("day_of_week")).alias("mode_day_of_week"))

        train_df = train_df_grouped_users \
                    .join(train_df_grouped, "account_id", how="inner") 

        train_df = train_df.withColumn("buy_prob", col("buys") / col("user_total_buys")).na.fill(0.0)
        train_df = train_df.withColumn("visit_prob", col("visits") / col("user_total_visits")).na.fill(0.0)

        train_df.write.parquet(self.output().path)


class PrepareIfoodSessionsDataFrames(BasePrepareDataFrames):

    def requires(self):
        return GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(
                    test_size=self.session_test_size, 
                    minimum_interactions=self.minimum_interactions,
                    sample_size=self.sample_size), \
               IndexAccountsAndMerchantsOfSessionTrainDataset(
                    test_size=self.session_test_size,
                    minimum_interactions=self.minimum_interactions,
                    sample_size=self.sample_size), \
                LoggingPolicyPsDataset(
                    test_size=self.session_test_size,
                    minimum_interactions=self.minimum_interactions,
                    sample_size=self.sample_size)                                                                  

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def stratification_property(self) -> str:
        return "buy"

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
        df    = pd.read_parquet(self.input()[1].path)
        df_ps = pd.read_csv(self.input()[2].path)

        for c in ["buy", "user_total_buys", "user_total_visits", "hist_buys", "hist_visits"]:
            df[c] = df[c].astype(float)
        
        #df["buy_prob"]   = df["hist_buys"].divide(df["user_total_buys"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        #df["visit_prob"] = df["hist_visits"].divide(df["user_total_visits"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df["n_users"]    = self.num_users
        df["n_items"]    = self.num_businesses

        # Join With LogginPolicyPS
        df = df.merge(df_ps, on=["account_idx", "merchant_idx"])

        return df

    @property
    def metadata_data_frame_path(self) -> str:
        return self.input()[0][1].path

class PrepareIfoodIntraSessionInteractionsDataFrames(BasePrepareDataFrames):

    def requires(self):
        return GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(
                        test_size=self.session_test_size, 
                        minimum_interactions=self.minimum_interactions,
                        sample_size=self.sample_size), \
               CreateIntraSessionInteractionDataset(
                        test_size=self.session_test_size,
                        minimum_interactions=self.minimum_interactions,
                        sample_size=self.sample_size)

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def stratification_property(self) -> str:
        return "binary_buys"

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
        df = pd.read_parquet(self.input()[1].path)
        df["merchant_idx"]  = df.merchant_idx_A
        df["binary_buys"]   = 1.0
        df["n_users"]       = self.num_users
        df["n_items"]       = self.num_businesses

        return df

    @property
    def metadata_data_frame_path(self) -> str:
        return self.input()[0][1].path



class PrepareIfoodInteractionsDataFrames(BasePrepareDataFrames):
    def requires(self):
        return GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(
                    test_size=self.session_test_size, 
                    minimum_interactions=self.minimum_interactions,
                    sample_size=self.sample_size), \
               CreateInteractionDataset(
                   test_size=self.session_test_size,
                   minimum_interactions=self.minimum_interactions,
                   sample_size=self.sample_size)

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def stratification_property(self) -> str:
        return "binary_buys"

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
        df = pd.read_parquet(self.input()[1].path)
        df["binary_buys"] = (df["buys"] > 0).astype(float)
        df['buys'] = df['buys'].astype(float)
        df['visits'] = df['visits'].astype(float)
        df["n_users"] = self.num_users
        df["n_items"] = self.num_businesses
        return df

    @property
    def metadata_data_frame_path(self) -> str:
        return self.input()[0][1].path


class PrepareIfoodVisitsBuysInteractionsDataFrames(PrepareIfoodInteractionsDataFrames):

    @property
    def stratification_property(self) -> str:
        pass

    def read_data_frame(self) -> pd.DataFrame:
        df = pd.read_parquet(self.input()[1].path)
        df["binary_buys"] = (df["buys"] > 0).astype(float)
        df['buys'] = df['buys'].astype(float)
        df['visits'] = df['visits'].astype(float)
        df["n_users"] = self.num_users
        df["n_items"] = self.num_businesses
        return df


class PrepareIfoodAccountMatrixWithBinaryBuysDataFrames(BasePrepareDataFrames):
    split_per_user: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(
                    test_size=self.session_test_size, 
                    minimum_interactions=self.minimum_interactions,
                    sample_size=self.sample_size), \
               CreateInteractionDataset(
                    test_size=self.session_test_size, 
                    minimum_interactions=self.minimum_interactions,
                    sample_size=self.sample_size)

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

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
        df = pd.read_parquet(self.input()[1].path)
        df["buys"] = (df["buys"] > 0).astype(float)
        if self.split_per_user:
            df = self._transform_data_frame(df)

        return df

    def transform_data_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.split_per_user:
            return self._transform_data_frame(df)
        return df


class PrepareIfoodMerchantMatrixWithBinaryBuysAndContentDataFrames(PrepareIfoodAccountMatrixWithBinaryBuysDataFrames):

    def _transform_data_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["buys"] > 0]
        df = df[["account_idx", "merchant_idx", "buys"]]
        df = df.groupby('account_idx')[['merchant_idx', 'buys']].apply(lambda x: x.values.tolist()).reset_index()
        df.columns = ["account_idx", "buys_per_merchant"]

        df["n_users"] = self.num_users
        df["n_items"] = self.num_businesses

        return df


class PrepareIfoodIndexedSessionTestData(BasePySparkTask):
    test_size: float = luigi.FloatParameter(default=0.10)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    def requires(self):
        return SplitSessionDataset(test_size=self.test_size, 
                                    minimum_interactions=self.minimum_interactions,
                                    sample_size=self.sample_size), \
               GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(test_size=self.test_size,
                                                                           minimum_interactions=self.minimum_interactions,
                                                                           sample_size=self.sample_size)

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR,
                                              "indexed_session_test_data_%.2f_k=%d_s=%d.parquet" % (self.test_size,
                                                                                                    self.minimum_interactions,
                                                                                                    self.sample_size)))

    def main(self, sc: SparkContext, *args):
        spark = SparkSession(sc)

        session_df = spark.read.parquet(self.input()[0][1].path)

        session_df = session_df.withColumn("mode_shift_idx", session_df.shift_idx)
        session_df = session_df.withColumn("mode_day_of_week", session_df.day_of_week)
        session_df = session_df.withColumn("visit", lit(1))

        account_df = spark.read.csv(self.input()[1][0].path, header=True, inferSchema=True)
        merchant_df = spark.read.csv(self.input()[1][1].path, header=True, inferSchema=True) \
            .select("merchant_idx", "merchant_id", "shifts", "days_of_week")

        session_df = session_df \
            .join(account_df, "account_id", how="inner") \
            .join(merchant_df, "merchant_id", how="inner") \
            .select("session_id", "click_timestamp",
                    "account_idx", "merchant_idx",
                    "shift_idx", "mode_shift_idx", 
                    "day_of_week", "mode_day_of_week",
                    "visit", 'buy').sort("click_timestamp")

        session_df.write.parquet(self.output().path)

class PrepareIfoodIndexedOrdersTestData(BasePySparkTask):
    test_size: float = luigi.FloatParameter(default=0.10)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    def requires(self):
        return SplitSessionDataset(test_size=self.test_size, 
                                    minimum_interactions=self.minimum_interactions,
                                    sample_size=self.sample_size), \
               GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(test_size=self.test_size,
                                                                           minimum_interactions=self.minimum_interactions,
                                                                           sample_size=self.sample_size)

    def output(self):
        return  luigi.LocalTarget(os.path.join(DATASET_DIR,
                                              "indexed_orders_test_data_%.2f_k=%d_s=%d.parquet" % (self.test_size,
                                                                                              self.minimum_interactions,
                                                                                              self.sample_size))) 

    def main(self, sc: SparkContext, *args):
        spark = SparkSession(sc)

        session_df  = spark.read.parquet(self.input()[0][1].path)
        print("session_df size: ", session_df.count())

        session_df   = session_df.withColumn("mode_shift_idx", session_df.shift_idx)
        session_df   = session_df.withColumn("mode_day_of_week", session_df.day_of_week)

        account_df   = spark.read.csv(self.input()[1][0].path, header=True, inferSchema=True)
        merchant_df  = spark.read.csv(self.input()[1][1].path, header=True, inferSchema=True) \
                                    .select("merchant_idx", "merchant_id", "shifts", "days_of_week")
        
        session_train  = spark.read.parquet(self.input()[0][0].path)
        session_eval   = spark.read.parquet(self.input()[0][1].path)

        count_visits   = session_df.filter(session_df.buy == 0).count()
        count_buys     = session_df.filter(session_df.buy == 1).count()

        #PrepareIfoodIndexedSessionTestData
        # Filter session with visits or not
        orders_df = session_df.filter(session_df.buy == 1)

        # Join with merchant the same caracteristics to simulate merchants open
        orders_df = orders_df.join(merchant_df, [merchant_df.shifts.contains(orders_df.shift),
                                                 merchant_df.days_of_week.contains(orders_df.day_of_week)]) \
                    .drop(merchant_df.merchant_id) \
                    .select(session_df.columns + ["merchant_idx"])
        

        # Group and similar merchants list (merchant_idx_list)
        orders_df = orders_df.groupBy(session_df.columns)\
                    .agg(collect_set("merchant_idx").alias("merchant_idx_list")) \
                    .drop("merchant_idx")


        orders_df = orders_df \
                    .withColumn("count_visits", lit(count_visits)) \
                    .withColumn("count_buys", lit(count_buys)) \
                    .join(account_df, "account_id", how="inner") \
                    .join(merchant_df, "merchant_id", how="inner") \
                    .select("session_id", "account_idx", "merchant_idx",
                            "merchant_idx_list", "shift", "shift_idx",
                            "mode_shift_idx", "mode_day_of_week",
                            "day_of_week", "count_buys", "count_visits", "buy").dropDuplicates()

        test_size = orders_df.count()
        
        if test_size == 0:
            raise Exception("Test Data Empty after filtered...")
        
        orders_df.write.parquet(self.output().path)


class ListAccountMerchantTuplesForIfoodIndexedOrdersTestData(BasePySparkTask):
    test_size: float = luigi.FloatParameter(default=0.10)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    def requires(self):
        return GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(
                    test_size=self.test_size, 
                    minimum_interactions=self.minimum_interactions,
                    sample_size=self.sample_size), \
               PrepareIfoodIndexedOrdersTestData(
                    test_size=self.test_size, 
                    minimum_interactions=self.minimum_interactions,
                    sample_size=self.sample_size)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(DATASET_DIR, "account_merchant_tuples_from_test_data_%.2f_k=%d_s=%d.parquet" % (
                self.test_size, self.minimum_interactions, self.sample_size)))

    def main(self, sc: SparkContext, *args):
        spark = SparkSession(sc)

        df = spark.read.parquet(self.input()[1].path)

        tuples_df = df.union(df.withColumn("merchant_idx", explode(df.merchant_idx_list))).drop("merchant_idx_list") \
            .dropDuplicates()

        tuples_df.write.parquet(self.output().path)
