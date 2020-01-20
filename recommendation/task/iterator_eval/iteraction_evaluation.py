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
import json
from recommendation.task.model.base import BaseTorchModelTraining
from recommendation.model.bandit import BanditPolicy, EpsilonGreedy, LinUCB, RandomPolicy, ModelPolicy, PercentileAdaptiveGreedy, AdaptiveGreedy
from recommendation.task.data_preparation.ifood import CheckDataset, GenerateIndicesForAccountsAndMerchantsDataset
from tqdm import tqdm
from recommendation.files import get_params, get_task_dir
from recommendation.task.data_preparation.base import BasePySparkTask
from pyspark.sql.functions import when, lit
from pyspark.sql.types import IntegerType
from recommendation.task.data_preparation.ifood import BaseDir
import pyspark.sql.functions as F
from tzlocal import get_localzone
import gc
import timeit

from recommendation.task.iterator_eval.base import BaseIterationEvaluation, BuildIteractionDatasetTask, MergeIteractionDatasetTask
from recommendation.task.ifood import EvaluateIfoodModel
from recommendation.task.evaluation import BaseEvaluationTask



# DATASET_PROCESSED_PATH="./output/ifood/dataset_5" PYTHONPATH="." nohup luigi \
# --module recommendation.task.iterator_eval.iteraction_evaluation IterationEvaluationTask \
# --local-scheduler --model-task-id=ContextualBanditsTraining_selu____512_0771f4fe24 \
# --model-module=recommendation.task.model.contextual_bandits \
# --model-cls=ContextualBanditsTraining --model-module-eval=recommendation.task.ifood   \
# --model-cls-eval=EvaluateIfoodFullContentModel \
# --run-type=reinforcement \
# --bandit-policy epsilon_greedy  \
# --bandit-policy-params '{"epsilon": 0.1}' \
# --batch-size 55000 > nohup_dt5 &
class IterationEvaluationTask(BaseIterationEvaluation): 

    @property
    def task_name(self):
        return self.model_task_id + "_" + self.task_id.split("_")[-1]

    def model_evaluate(self, params) -> EvaluateIfoodModel:
        module = importlib.import_module(self.model_module_eval)
        class_ = getattr(module, self.model_cls_eval)

        self._model_evaluate = class_(**{'model_module': self.model_module, 
                                        'model_cls': self.model_cls,
                                        'bandit_policy': self.bandit_policy,
                                        'bandit_policy_params': self.bandit_policy_params,
                                        'nofilter_iteractions_test':  self.is_reinforcement,
                                        'no_offpolicy_eval': self.no_offpolicy_eval,
                                        'task_hash': self.task_id}, 
                                        **params)
                          
        return self._model_evaluate     

        
    def run(self):
        os.makedirs(os.path.split(self.output()[0].path)[0], exist_ok=True)

        logs         = []
        full_df      = pd.read_parquet(self.input().path).sort_values("click_timestamp")

        # set new info_session dataset
        os.environ['DATASET_INFO_SESSION'] = self.input().path

        path_bandit_weights = 'none'

        # each batch
        # i=n, sample-size = batch * (n + 1), session-test-size = sample-size / ( n + 1)
        for i in range(math.ceil(len(full_df)/self.batch_size)):
            start = timeit.default_timer()

            sample_size           = self.batch_size * (i + 1)
            test_size             = 1 / (i + 1)  
            minimum_interactions  = self.minimum_interactions

            #  If reinforcement the first interaction dont have train data
            #  fix test_size and add only necessery to run (5 examples)
            if i == 0:
              if self.is_reinforcement:
                minimum_interactions  = 0
                test_size             = test_size - (5/sample_size)
              else: continue


            task_train   = self.model_training({'sample_size': sample_size, 
                                                'session_test_size': test_size, 
                                                'minimum_interactions': minimum_interactions})

            task_eval    = self.model_evaluate({'model_task_id': task_train.task_id,
                                                'bandit_weights': path_bandit_weights})

            # Train Model
            yield task_train

            # Evalution Model
            yield task_eval
            
            if self.is_reinforcement:

                task_merge   = MergeIteractionDatasetTask(batch_size=self.batch_size,
                                                          test_size=test_size, 
                                                          sample_size=sample_size,
                                                          minimum_interactions=minimum_interactions,
                                                          evaluation_path=task_eval.output_path)                

                # Merge Dataset
                yield task_merge
                # New bandit save
                path_bandit_weights = task_eval.bandit_path

                # set new info_session dataset
                os.environ['DATASET_INFO_SESSION'] = task_merge.output_path
                

            logs.append({'i': i,
                        'time': int((timeit.default_timer() - start)/60),
                        'model_task_id': self.model_task_id,
                        'train_path':   task_train.output().path, 
                        'eval_path':    task_eval.output_path, 
                        'bandit_path':  task_eval.bandit_path,
                        'sample_size':  sample_size,
                        'test_size':    test_size})

            # Save logs
            self.save_logs(logs)    
            
            task_train.cache_cleanup()
            del task_train
            del task_eval
            del task_merge     
            #task_eval.cache_cleanup()
            gc.collect()

        # params.json
        with open(self.output()[1].path, "w") as params_file:
            json.dump(self.param_kwargs, params_file, default=lambda o: dict(o), indent=4)


# DATASET_PROCESSED_PATH="./output/ifood/dataset_6" PYTHONPATH="." luigi \
# --module recommendation.task.iterator_eval.iteraction_evaluation IterationEvaluationWithoutModelTask \
# --local-scheduler \
# --model-module-eval recommendation.task.ifood   \
# --model-cls-eval EvaluateMostPopularPerUserIfoodModel \
# --run-type=reinforcement --bandit-policy epsilon_greedy  \
#   --bandit-policy-params '{"epsilon": 0.1}' --batch-size 55000
class IterationEvaluationWithoutModelTask(BaseIterationEvaluation): #WrapperTask
    model_task_id: str = luigi.Parameter(default='none')
    model_module: str = luigi.Parameter(default="none")
    model_cls: str = luigi.Parameter(default="none")
    model_module_eval: str = luigi.Parameter(default="recommendation.task.ifood")    
    model_cls_eval: str = luigi.Parameter(default="EvaluateMostPopularPerUserIfoodModel")
    
    @property
    def task_name(self):
        return self.model_cls_eval + "_" + self.task_id.split("_")[-1]

    def model_evaluate(self, params) -> BaseEvaluationTask:
        module = importlib.import_module(self.model_module_eval)
        class_ = getattr(module, self.model_cls_eval)

        eval_params = {**{'model_module': self.model_module, 
                          'model_cls': self.model_cls,
                          'bandit_policy': self.bandit_policy,
                          'bandit_policy_params': self.bandit_policy_params,
                          'nofilter_iteractions_test':  self.is_reinforcement,
                          'no_offpolicy_eval': self.no_offpolicy_eval,                          
                          'task_hash': self.task_id}, **params}

        self._model_evaluate = class_(**eval_params)

        return self._model_evaluate     

    def run(self):
        os.makedirs(os.path.split(self.output()[0].path)[0], exist_ok=True)

        logs         = []
        full_df      = pd.read_parquet(self.input().path).sort_values("click_timestamp")

        # set new info_session dataset
        os.environ['DATASET_INFO_SESSION'] = self.input().path

        path_bandit_weights = 'none'

        # each batch
        # i=n, sample-size = batch * (n + 1), session-test-size = sample-size / ( n + 1)
        for i in range(math.ceil(len(full_df)/self.batch_size)):
            start = timeit.default_timer()

            sample_size           = self.batch_size * (i + 1)
            test_size             = 1 / (i + 1)  
            minimum_interactions  = self.minimum_interactions

            #  If reinforcement the first interaction dont have train data
            #  fix test_size and add only necessery to run (5 examples)
            if i == 0:
              if self.is_reinforcement:
                minimum_interactions  = 0
                test_size             = test_size - (5/sample_size)
              else: continue

            task_eval    = self.model_evaluate({'sample_size': sample_size, 
                                                'test_size': test_size, 
                                                'minimum_interactions': minimum_interactions,
                                                'bandit_weights': path_bandit_weights})

            # Evalution Model
            yield task_eval
            
            if self.is_reinforcement:
                task_merge   = MergeIteractionDatasetTask(batch_size=self.batch_size,
                                                          test_size=test_size, 
                                                          sample_size=sample_size,
                                                          minimum_interactions=minimum_interactions,
                                                          evaluation_path=task_eval.output_path)

                # Merge Dataset
                yield task_merge

                # New bandit save
                path_bandit_weights = task_eval.bandit_path

                # set new info_session dataset
                os.environ['DATASET_INFO_SESSION'] = task_merge.output_path
            
           
            logs.append({'i': i,
                        'time': int((timeit.default_timer() - start)/60),
                        'model_task_id': self.model_cls_eval,
                        'train_path':   "", 
                        'eval_path':    task_eval.output_path, 
                        'sample_size':  sample_size,
                        'sample_percent': len(full_df)/sample_size,
                        'test_size':    test_size})
            #del task_eval

            # Save logs
            self.save_logs(logs)    
            gc.collect()

        # params.json
        with open(self.output()[1].path, "w") as params_file:
            json.dump(self.param_kwargs, params_file, default=lambda o: dict(o), indent=4)
