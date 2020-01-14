import abc
import importlib
import os
import luigi
import json
import pandas as pd
from typing import Dict, Tuple, List, Any, Type, Union
from torch.utils.data.dataset import Dataset
from pyspark import SparkContext
from pyspark.sql import SparkSession
from recommendation.task.model.base import BaseTorchModelTraining, load_torch_model_training_from_task_id
from recommendation.task.evaluation import BaseEvaluationTask
from recommendation.model.bandit import BanditPolicy, EpsilonGreedy, LinUCB, RandomPolicy, ModelPolicy
from recommendation.task.data_preparation.ifood import GlobalConfig, SplitSessionDataset, CheckDataset, PrepareIfoodSessionsDataFrames, GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset
from recommendation.utils import chunks, parallel_literal_eval
from tqdm import tqdm
import math
import shutil  
from recommendation.files import get_params_path, get_weights_path, get_params, get_history_path, \
    get_tensorboard_logdir, get_task_dir

from recommendation.torch import NoAutoCollationDataLoader
from recommendation.task.data_preparation.base import BasePySparkTask, BasePrepareDataFrames, BaseDownloadDataset
from luigi.contrib.external_program import ExternalProgramTask, ExternalPythonProgramTask
from luigi.contrib.external_program import ExternalProgramRunError
from recommendation.task.data_preparation.base import BasePySparkTask
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, lit
from pyspark.sql.types import IntegerType

import pyspark.sql.functions as F
from tzlocal import get_localzone
LOCAL_TZ: str = str(get_localzone())
BASE_DIR: str = os.path.join("output", "ifood")

_BANDIT_POLICIES: Dict[str, Type[BanditPolicy]] = dict(epsilon_greedy=EpsilonGreedy, lin_ucb=LinUCB, random=RandomPolicy, model=ModelPolicy, none=None)


# PYTHONPATH="." luigi \
# --module recommendation.task.iterator_eval.base IterationEvaluationTask \
# --local-scheduler \
# --model-task-id=ContextualBanditsTraining_selu____512_b3940c3ec7 \
# --model-module=recommendation.task.model.contextual_bandits \
# --model-cls=ContextualBanditsTraining \
# --model-module-eval=recommendation.task.ifood   \
# --model-cls-eval=EvaluateIfoodFullContentModel 
class IterationEvaluationTask(luigi.Task): #WrapperTask
    model_module: str = luigi.Parameter(default="recommendation.task.model.contextual_bandits")
    model_cls: str = luigi.Parameter(default="ContextualBanditsTraining")
    model_module_eval: str = luigi.Parameter(default="recommendation.task.ifood")    
    model_cls_eval: str = luigi.Parameter(default="EvaluateIfoodFullContentModel")
    model_task_id: str = luigi.Parameter()
    bandit_policy: str = luigi.ChoiceParameter(choices=_BANDIT_POLICIES.keys(), default="none")
    bandit_policy_params: Dict[str, Any] = luigi.DictParameter(default={})
    
    batch_size: int = luigi.IntParameter(default = 450000)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    def requires(self):
        return SplitSessionDataset(test_size=0, sample_size=-1, minimum_interactions=0)

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

        self._model_training = class_(**train_params)

        return self._model_training

    def model_evaluate(self, task_id) -> BaseEvaluationTask:
        module = importlib.import_module(self.model_module_eval)
        class_ = getattr(module, self.model_cls_eval)

        self._model_evaluate = class_(**{'model_module': self.model_module, 
                                        'model_cls': self.model_cls,
                                        'model_task_id': task_id,
                                        'bandit_policy': self.bandit_policy,
                                        'bandit_policy_params': self.bandit_policy_params})

        return self._model_evaluate     

    def save_logs(self, log):
        # params.json
        with open(self.output()[1].path, "w") as params_file:
            json.dump(self.param_kwargs, params_file, default=lambda o: dict(o), indent=4)

        # Save logs
        df = pd.DataFrame(log)
        print(df.head())
        df.to_csv(self.output()[0].path)
        

    def run(self):
        os.makedirs(os.path.split(self.output()[0].path)[0], exist_ok=True)

        logs         = []
        full_df      = pd.read_parquet(self.input()[0].path).sort_values("click_timestamp")

        # each batch
        # i=n, sample-size = batch * (n + 1), session-test-size = sample-size / ( n + 1)
        for i in range(math.ceil(len(full_df)/self.batch_size)):
            if i == 0:
                continue

            sample_size  = self.batch_size * (i + 1)
            test_size    = 1 / (i + 1)
            

            task_train   = self.model_training({'sample_size': sample_size, 
                                                'session_test_size': test_size, 
                                                'minimum_interactions': self.minimum_interactions})

            task_eval    = self.model_evaluate(task_id = task_train.task_id)

            task_merge   = MergeSessionDatasetTask(test_size=test_size, 
                                                    sample_size=sample_size,
                                                    minimum_interactions=self.minimum_interactions,
                                                    evaluation_path=task_eval.output_path)

            yield GlobalConfig(path_info_session="sdsdsd")
            #'CheckDataset_path_info_session': 'dddddddddddd'
            # Train Model
            yield task_train

            # Evalution Model
            yield task_eval

            # Merge Dataset
            yield task_merge

            logs.append({'i': i,
                    'train_path':   task_train.output().path, 
                    'eval_path':    task_eval.output_path, 
                    'sample_size':  sample_size,
                    'test_size':    test_size})

        # Save logs
        self.save_logs(logs)    

# PYTHONPATH="." luigi \
# --module recommendation.task.iterator_eval.base IterationEvaluationWithoutModelTask \
# --local-scheduler \
# --model-module-eval=recommendation.task.ifood   \
# --model-cls-eval=EvaluateMostPopularPerUserIfoodModel 
class IterationEvaluationWithoutModelTask(IterationEvaluationTask): #WrapperTask
    model_task_id: str = luigi.Parameter(default='none')

    @property
    def task_name(self):
        return self.model_cls_eval + "_" + self.task_id.split("_")[-1]

    def model_evaluate(self, sample_size, test_size) -> BaseEvaluationTask:
        module = importlib.import_module(self.model_module_eval)
        class_ = getattr(module, self.model_cls_eval)

        self._model_evaluate = class_(**{'model_module': self.model_module, 
                                        'model_cls': self.model_cls,
                                        'sample_size': sample_size,
                                        'test_size': test_size,
                                        'bandit_policy': self.bandit_policy,
                                        'bandit_policy_params': self.bandit_policy_params})

        return self._model_evaluate     

    def run(self):
        os.makedirs(os.path.split(self.output()[0].path)[0], exist_ok=True)

        logs         = []
        full_df      = pd.read_parquet(self.input()[0].path).sort_values("click_timestamp")

        # each batch
        # i=n, sample-size = batch * (n + 1), session-test-size = sample-size / ( n + 1)
        for i in range(math.ceil(len(full_df)/self.batch_size)):
            if i == 0:
                continue

            sample_size  = self.batch_size * (i + 1)
            test_size    = 1 / (i + 1)
            
            task_eval    = self.model_evaluate(sample_size = sample_size, test_size = test_size)

            yield task_eval

            #l = self.run_batch(i, sample_size, test_size)
            logs.append({'i': i,
                    'train_path':   "", 
                    'eval_path':    task_eval.output_path, 
                    'sample_size':  sample_size,
                    'test_size':    test_size})
        # Save logs
        self.save_logs(logs)  

# PYTHONPATH="." luigi \
# --module recommendation.task.iterator_eval.base MergeSessionDatasetTask \
# --local-scheduler \
# --evaluation-path='output/evaluation/EvaluateAutoEncoderIfoodModel/results/VariationalAutoEncoderTraining_selu____500_fc62ac744a_6534ac1232'  
class MergeSessionDatasetTask(BasePySparkTask):
    test_size: float = luigi.FloatParameter()
    sample_size: int = luigi.IntParameter()
    minimum_interactions: int = luigi.FloatParameter()
    evaluation_path: str = luigi.Parameter()

    def requires(self):
        return CheckDataset(), \
                GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(
                    sample_size=self.sample_size,
                    minimum_interactions=self.minimum_interactions,
                    test_size=self.test_size
                )

    def output(self):
        return luigi.LocalTarget(self.output_path)

    @property
    def output_path(self):
        return os.path.join(BASE_DIR, 'ufg_dataset_all', "info_session_merged", self.task_id)

    def main(self, sc: SparkContext, *args):
        #os.makedirs(self.output_path, exist_ok=True)
        spark = SparkSession(sc)

        # Load info_session
        df_info_session = spark.read.parquet(self.input()[0][6].path)

        # Load merchant indices
        df_mearchant = spark.read.csv(self.input()[1][1].path,header=True, inferSchema=True)
        df_mearchant = df_mearchant.select('merchant_idx', 'merchant_id')

        # Load evaluation dataset
        #'../output/evaluation/EvaluateAutoEncoderIfoodModel/results/VariationalAutoEncoderTraining_selu____500_fc62ac744a_6534ac1232/orders_with_metrics.csv'
        df_eval = spark.read.csv(os.path.join(self.evaluation_path, 'orders_with_metrics.csv'), header=True, inferSchema=True).sort('click_timestamp')
        df_eval = df_eval.withColumn("click_timestamp", F.from_utc_timestamp(df_eval.click_timestamp, LOCAL_TZ))
        df_eval = df_eval.join(df_mearchant, df_eval.rhat_merchant_idx == df_mearchant.merchant_idx)
        df_eval = df_eval.withColumnRenamed("merchant_id", "rhat_merchant_id")
        df_eval = df_eval.withColumn("buy", lit(1))

        # Join order interactions
        df = df_info_session.join(
            df_eval.select('session_id', 'account_id', 'click_timestamp', 'rhat_merchant_id', 'buy'), 
            ['session_id', 'account_id', 'click_timestamp', 'buy'], how='left').cache()

        # Rewrite original interaction 
        df = df.withColumn('buy', when(df.rhat_merchant_id.isNull(), df.buy)\
                .otherwise((df.merchant_id == df.rhat_merchant_id).cast(IntegerType())))
        df = df.withColumn('merchant_id', when(df.rhat_merchant_id.isNull(), df.merchant_id)\
                .otherwise(df.rhat_merchant_id))

        df.select(df_info_session.columns).write.mode("overwrite").parquet(self.output().path)


