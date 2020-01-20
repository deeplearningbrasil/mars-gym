import importlib
import os
import luigi
import pandas as pd
from typing import Dict, Tuple, List, Any, Type, Union
from torch.utils.data.dataset import Dataset
from pyspark import SparkContext
from pyspark.sql import SparkSession
import random
import math
from recommendation.task.model.base import BaseTorchModelTraining
from recommendation.model.bandit import BanditPolicy, EpsilonGreedy, LinUCB, RandomPolicy, ModelPolicy, PercentileAdaptiveGreedy, AdaptiveGreedy
from recommendation.task.data_preparation.ifood import CheckDataset, CleanSessionDataset, GenerateIndicesForAccountsAndMerchantsDataset
from tqdm import tqdm
from recommendation.files import get_task_dir, get_task_dir
from recommendation.task.data_preparation.base import BasePySparkTask
from pyspark.sql.functions import when, lit
from pyspark.sql.types import IntegerType
from recommendation.task.data_preparation.ifood import BaseDir
import pyspark.sql.functions as F
from recommendation.files import get_params, get_task_dir

from tzlocal import get_localzone
import gc

LOCAL_TZ: str = str(get_localzone())
_BANDIT_POLICIES: Dict[str, Type[BanditPolicy]] = dict(epsilon_greedy=EpsilonGreedy, lin_ucb=LinUCB, random=RandomPolicy, \
    percentile_adaptive=PercentileAdaptiveGreedy, adaptive=AdaptiveGreedy, model=ModelPolicy, none=None)

class BaseIterationEvaluation(luigi.Task):
    run_type: str = luigi.ChoiceParameter(choices=["supervised", 'reinforcement'], default="supervised")

    model_module: str = luigi.Parameter(default="recommendation.task.model.contextual_bandits")
    model_cls: str = luigi.Parameter(default="ContextualBanditsTraining")
    model_module_eval: str = luigi.Parameter(default="recommendation.task.ifood")    
    model_cls_eval: str = luigi.Parameter(default="EvaluateIfoodFullContentModel")
    model_task_id: str = luigi.Parameter()
    bandit_policy: str = luigi.ChoiceParameter(choices=_BANDIT_POLICIES.keys(), default="none")
    bandit_policy_params: Dict[str, Any] = luigi.DictParameter(default={})
    
    batch_size: int = luigi.IntParameter(default = 750000) # 750000 | 165000
    minimum_interactions: int = luigi.FloatParameter(default=5)
    no_offpolicy_eval: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return BuildIteractionDatasetTask(run_type = self.run_type)

    def output(self):
        return  luigi.LocalTarget(os.path.join("output", "evaluation", self.__class__.__name__, 
                                    "results", self.task_name, "history.csv")), \
                luigi.LocalTarget(os.path.join("output", "evaluation", self.__class__.__name__, 
                                    "results", self.task_name, "params.json"))          

    @property
    def task_name(self):
        return self.model_task_id + "_" + self.task_id.split("_")[-1]

    def model_training(self, params) -> BaseTorchModelTraining:
        module   = importlib.import_module(self.model_module)
        class_   = getattr(module, self.model_cls)
        task_dir = get_task_dir(class_, self.model_task_id)

        train_params = get_params(task_dir)
        train_params = {**train_params, **params}
        train_params['task_hash'] = self.task_id

        self._model_training = class_(**train_params)

        return self._model_training

    def save_logs(self, log):
        # Save logs
        df = pd.DataFrame(log)
        print(df.head())
        df.to_csv(self.output()[0].path, index=False)     

    @property
    def seed(self):
        if not hasattr(self, "_seed"):
            self._seed = random.randint(0, 1000) if self.is_reinforcement else 42
        return self._seed        

    @property
    def is_reinforcement(self):
        return self.run_type == "reinforcement"    

class BuildIteractionDatasetTask(BasePySparkTask):
    run_type: str = luigi.ChoiceParameter(choices=["supervised", 'reinforcement'], default="supervised")

    # GroundTruth Dataset
    #
    def requires(self):
        return CheckDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join(BaseDir().dataset_processed, "info_session", "ground_truth"))

    def main(self, sc: SparkContext, *args):
        #os.makedirs(os.path.join(BaseDir().dataset_processed, exist_ok=True)
        spark = SparkSession(sc)
        
        df = spark.read.parquet(self.input()[6].path)

        if self.run_type == "reinforcement":
            df = df.filter(df.buy == 1)

        df.write.parquet(self.output().path)

# PYTHONPATH="." luigi \
# --module recommendation.task.iterator_eval.base MergeIteractionDatasetTask \
# --local-scheduler \
# --evaluation-path='output/evaluation/EvaluateAutoEncoderIfoodModel/results/VariationalAutoEncoderTraining_selu____500_fc62ac744a_6534ac1232'  
class MergeIteractionDatasetTask(BasePySparkTask):
    test_size: float = luigi.FloatParameter()
    sample_size: int = luigi.IntParameter()
    minimum_interactions: int = luigi.FloatParameter()
    evaluation_path: str = luigi.Parameter()
    batch_size: int = luigi.IntParameter() # 750000 | 165000

    def requires(self):
        return  BuildIteractionDatasetTask(),\
                CheckDataset(), \
                GenerateIndicesForAccountsAndMerchantsDataset(sample_size=self.sample_size)

    def output(self):
        return luigi.LocalTarget(self.output_path)

    @property
    def output_path(self):
       return os.path.join(BaseDir().dataset_processed, "info_session", "merged", self.task_id)

    def merge_real_test_with_evaluation_data(self, spark, df_real, df_eval):
        df = df_real.cache()
        #df = df_real.withColumn("buy", lit(0))

        # Load merchant indices idx
        df_mearchant        = spark.read.csv(self.input()[2][1].path,header=True, inferSchema=True)
        df_mearchant        = df_mearchant.select('merchant_idx', 'merchant_id')
        df_rhat_mearchant   = df_mearchant.withColumnRenamed("merchant_id", "rhat_merchant_id")\
                                            .withColumnRenamed("merchant_idx", "rhat_merchant_idx")

        df_eval = df_eval.join(df_mearchant, 'merchant_idx', how='inner')
        df_eval = df_eval.join(df_rhat_mearchant,'rhat_merchant_idx' , how='inner')\
                            .withColumn("buy", lit(1))

        # Join order interactions
        df = df.join(
                df_eval.select('session_id', 'account_id', 'merchant_id', 'dt_partition', 'rhat_merchant_id', 'buy'), 
                ['session_id', 'account_id', 'merchant_id', 'dt_partition', 'buy'], how='left')

        df = df.withColumn("old_buy", df.buy)
        df = df.withColumn("old_merchant_id", df.merchant_id)

        # Rewrite original interaction 
        df = df.withColumn('buy', when(df.rhat_merchant_id.isNull(), 0)\
                .otherwise((df.merchant_id == df.rhat_merchant_id).cast(IntegerType())))
        df = df.withColumn('merchant_id', when(df.rhat_merchant_id.isNull(), df.merchant_id)\
                .otherwise(df.rhat_merchant_id))

        #df_info_session.columns
        #columns = ['session_id','account_id','merchant_id','click_timestamp','buy','dt_partition','old_buy', 'old_merchant_id']
        
        return df.select(df_real.columns)

#        columns = ['session_id','account_id','merchant_id','click_timestamp','buy','dt_partition','old_buy', 'old_merchant_id']
 #       df.select(columns).write.mode("overwrite").parquet(self.output().path)

    def main(self, sc: SparkContext, *args):
        #os.makedirs(self.output_path, exist_ok=True)
        spark = SparkSession(sc)

        # Load info_session
        df_groud_truth       = spark.read.parquet(self.input()[0].path)

        # Load info_session
        df_current_dataset   = spark.read.parquet(os.environ['DATASET_INFO_SESSION']).sort("click_timestamp")

        n_test      = math.ceil(self.test_size * self.sample_size)
        df_trained  = df_current_dataset.limit(self.sample_size).sort("click_timestamp").limit(self.sample_size - n_test)
        df_test_gt  = df_current_dataset.limit(self.sample_size).sort("click_timestamp", ascending=False).limit(n_test)

        # Join test groud truth with evaluation information
        #'../output/evaluation/EvaluateAutoEncoderIfoodModel/results/VariationalAutoEncoderTraining_selu____500_fc62ac744a_6534ac1232/orders_with_metrics.csv'
        df_eval      = spark.read.csv(os.path.join(self.evaluation_path, 'orders_with_metrics.csv'), header=True, inferSchema=True)
        test_eval_df = self.merge_real_test_with_evaluation_data(spark, df_test_gt, df_eval)

        # Filter next test set with groud truth information
        df_next_test_gt = df_groud_truth.sort("click_timestamp").limit(self.sample_size + self.batch_size)
        df_next_test_gt = df_next_test_gt.sort("click_timestamp", ascending=False).limit(self.batch_size)

        columns         = df_groud_truth.columns
        df_next_dataset = df_trained.select(columns)\
                            .union(test_eval_df.select(columns))\
                            .union(df_next_test_gt.select(columns)).sort("click_timestamp")


        df_next_dataset.write.mode("overwrite").parquet(self.output().path)