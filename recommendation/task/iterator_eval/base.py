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
from recommendation.task.data_preparation.ifood import SplitSessionDataset, CheckDataset, PrepareIfoodSessionsDataFrames
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

    def model_training(self, sample_size, test_size) -> BaseTorchModelTraining:
        module   = importlib.import_module(self.model_module)
        class_   = getattr(module, self.model_cls)
        task_dir = get_task_dir(class_, self.model_task_id)

        train_params = get_params(task_dir)

        train_params['sample_size'] = sample_size
        train_params['session_test_size'] = test_size

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

    def _save_params(self):
        with open(self.output()[1].path, "w") as params_file:
            json.dump(self.param_kwargs, params_file, default=lambda o: dict(o), indent=4)

    def run(self):
        os.makedirs(os.path.split(self.output()[0].path)[0], exist_ok=True)

        log          = []
        full_df      = pd.read_parquet(self.input()[0].path).sort_values("click_timestamp")

        # each batch
        # i=n, sample-size = batch * (n + 1), session-test-size = sample-size / ( n + 1)
        for i in range(math.ceil(len(full_df)/self.batch_size)):
            if i == 0:
                continue

            sample_size  = self.batch_size * (i + 1)
            test_size    = 1 / (i + 1)
            
            task_train   = self.model_training(sample_size = sample_size, test_size = test_size)
            task_eval    = self.model_evaluate(task_id = task_train.task_id)

            yield task_train
            yield task_eval

            log.append({'i': i,
                        'train_path':   task_train.output().path, 
                        'eval_path':    task_eval.output_path, 
                        'sample_size':  sample_size,
                        'test_size':    test_size})

        # Save logs
        df = pd.DataFrame(log)
        df.to_csv(self.output()[0].path)
        self._save_params()

        print(df.head())        