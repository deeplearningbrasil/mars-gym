import abc
import importlib
import os
import luigi
import json
import pandas as pd
from recommendation.files import get_params_path
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
# --module recommendation.task.iterator_eval.base BaseIteratorEvaluationTask \
# --local-scheduler \
# --model-task-id=ContextualBanditsTraining_selu____512_54d17b4f72 \
# --model-module=recommendation.task.model.contextual_bandits \
# --model-cls=ContextualBanditsTraining \
# --model-module-eval=recommendation.task.ifood   \
# --model-cls-eval=EvaluateIfoodFullContentModel 
# --bandit-policy: str = luigi.ChoiceParameter(choices=_BANDIT_POLICIES.keys(), default="none")
# --bandit-policy-params: Dict[str, Any] = luigi.DictParameter(default={})

class BaseIteratorEvaluationTask(luigi.Task, metaclass=abc.ABCMeta):
    model_module: str = luigi.Parameter(default="recommendation.task.model.contextual_bandits")
    model_cls: str = luigi.Parameter(default="ContextualBanditsTraining")
    model_module_eval: str = luigi.Parameter(default="recommendation.task.ifood")    
    model_cls_eval: str = luigi.Parameter(default="EvaluateIfoodFullContentModel")
    model_task_id: str = luigi.Parameter()
    bandit_policy: str = luigi.ChoiceParameter(choices=_BANDIT_POLICIES.keys(), default="none")
    bandit_policy_params: Dict[str, Any] = luigi.DictParameter(default={})
    batch_size: int = luigi.Parameter(default = 50)
    

    def requires(self):
       return (PrepareIfoodSessionsDataFrames(session_test_size=0, minimum_interactions=0),) + (self.model_training.requires(),)

                #,  self.model_evaluate.requires()
       #return self.model_evaluate.requires()

    @property
    def task_name(self):
        return self.model_task_id + "_" + self.task_id.split("_")[-1]

    @property
    def model_training(self) -> BaseTorchModelTraining:
        if not hasattr(self, "_model_training"):
            module = importlib.import_module(self.model_module)
            class_ = getattr(module, self.model_cls)

            self._model_training = load_torch_model_training_from_task_id(class_, self.model_task_id)

        return self._model_training

    @property
    def model_evaluate(self) -> BaseEvaluationTask:
      if not hasattr(self, "_model_evaluate"):
          module = importlib.import_module(self.model_module_eval)
          class_ = getattr(module, self.model_cls_eval)

          self._model_evaluate = class_(**{'model_module': self.model_module, 
                                          'model_cls': self.model_cls,
                                          'model_task_id': self.model_task_id})

      return self._model_evaluate      

    @property
    def n_items(self):
        return self.model_training.n_items

    @property
    def output_path(self):
        return os.path.join("output", "evaluation", self.__class__.__name__, "results", self.task_name)

    @property
    def train_data_frame_path(self) -> str:
        return self.input()[0][0].path

    @property
    def val_data_frame_path(self) -> str:
        return self.input()[0][1].path

    @property
    def test_data_frame_path(self) -> str:
        return self.input()[0][2].path

    @property
    def dataset(self) -> Dataset:
        if not hasattr(self, "_dataset"):
            print("Reading tuples files...")
            if self.model_training.project_config.output_column.name not in self.test_data_frame.columns:
                self.test_data_frame[self.model_training.project_config.output_column.name] = 1
            for auxiliar_output_column in self.model_training.project_config.auxiliar_output_columns:
                if auxiliar_output_column.name not in self.test_data_frame.columns:
                    self.test_data_frame[auxiliar_output_column.name] = 0

            self._dataset = self.model_training.project_config.dataset_class(self.test_data_frame,
                                                                             self.model_training.metadata_data_frame,
                                                                             self.model_training.project_config,
                                                                             negative_proportion=0.0)

        return self._dataset

    def _evaluate_account_merchant_tuples(self) -> Dict[Tuple[int, int], float]:
        generator = NoAutoCollationDataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                              num_workers=self.model_training.generator_workers,
                                              pin_memory=self.pin_memory if self.model_training.device == "cuda" else False)

        print("Loading trained model...")
        module = self.model_training.get_trained_module()

        trial = Trial(module,
                      criterion=lambda *args:
                      torch.zeros(1, device=self.model_training.torch_device, requires_grad=True)) \
            .with_test_generator(generator).to(self.model_training.torch_device)
        model_output: Union[torch.Tensor, Tuple[torch.Tensor]] = trial.predict(verbose=2)

        # TODO else with batch size
        scores_tensor: torch.Tensor = model_output if isinstance(model_output, torch.Tensor) else model_output[0][0]
        scores: np.ndarray = scores_tensor.detach().cpu().numpy()

        scores = self._transform_scores(scores)

        return self._create_dictionary_of_scores(scores)


    def run(self):
        print("==================")
        print(self.train_data_frame_path)
        df_dataset = pd.read_csv(self.train_data_frame_path).sort_values('click_timestamp')
        print(df_dataset.head())
        print(df_dataset.shape)

        scores_per_tuple     = self._evaluate_account_merchant_tuples()

#        print(self.model_training.train_data_frame_path)
#        print("==================")

        
        #df_train   = pd.read_csv(self.model_training.train_data_frame_path)

        #print(df_train.head())
        #print(df_train.shape)

        for indices in tqdm(chunks(range(len(df_dataset)), self.batch_size),
                            total=math.ceil(len(df_dataset) / self.batch_size)):
            rows: pd.DataFrame = df_dataset.iloc[indices]
            
            tmp_dataset = self.build_temp_dataset(rows)
            rec_list    = self.recommender()

            break


class SplitSessionDatasetTask(BasePySparkTask):
    batch_size: int = luigi.IntParameter(default = 730000)

    def requires(self):
        return CheckDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join("output", "ifood", "dataset_aux", "info_sessions"))

    def main(self, sc: SparkContext, *args):
        os.makedirs(os.path.split(self.output().path)[0], exist_ok=True)
        spark = SparkSession(sc)

        df     = spark.read.parquet(self.input()[6].path)
        count  = df.count()
        #print("Total Dataset ", count)
        #print("Total Partitions ", int(count/self.batch_size))

        i = 0
        for indices in tqdm(chunks(range(count), self.batch_size),
                            total=math.ceil(count / self.batch_size)):
            
            n_train  = (i*self.batch_size) + self.batch_size
            train_df = df.sort("click_timestamp").limit(n_train).cache()
            count    = train_df.count()

            train_df.write.parquet(os.path.join(self.output().path, "{}".format(i)))
            i = i+1


class ResetDatasetTask(luigi.Task):
    batch: int = luigi.Parameter(default = 0)
    
    def requires(self):
        return CheckDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join("output", "ifood", "dataset_aux", "reset_{}.txt".format(self.batch)))

    def run(self):
        shutil.rmtree(os.path.join(BASE_DIR, "dataset"), ignore_errors=True)

        shutil.move(os.path.join(BASE_DIR, "ufg_dataset_all", "info_session"), 
            os.path.join(BASE_DIR, "ufg_dataset_all", "info_session_bkp"))

        shutil.move(os.path.join(BASE_DIR, "dataset_aux", "info_sessions", str(i)), os.path.join(BASE_DIR, "ufg_dataset_all", "info_session"))

        with open(self.output().path, "w") as f:
            f.write(".")
        

class RunEvaluationTask(luigi.Task):
    def requires(self):
        return CheckDataset()

    def run(self):
        #shutil.rmtree(os.path.join(BASE_DIR, "dataset"), ignore_errors=True)

        os.makedirs(os.path.split(self.output().path)[0], exist_ok=True)

# PYTHONPATH="." luigi \
# --module recommendation.task.iterator_eval.base IteratorEvaluationTask \
# --local-scheduler \
# --model-task-id=ContextualBanditsTraining_selu____512_b3940c3ec7 \
# --model-module=recommendation.task.model.contextual_bandits \
# --model-cls=ContextualBanditsTraining \
# --model-module-eval=recommendation.task.ifood   \
# --model-cls-eval=EvaluateIfoodFullContentModel 
class IteratorEvaluationTask(luigi.Task): #WrapperTask
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
        return luigi.LocalTarget(os.path.join("output", "evaluation", self.__class__.__name__, "results", self.task_name, "log.csv"))

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
                                        'model_task_id': task_id})

        return self._model_evaluate     

    def run(self):
        log          = []
        full_df      = pd.read_parquet(self.input()[0].path).sort_values("click_timestamp")

        # each batch
        # i=n, sample-size = batch * (n + 1), session-test-size = sample-size / ( n + 1)
        for i in range(int(len(full_df)/self.batch_size)):
            if i == 0:
                continue

            sample_size  = self.batch_size * (i + 1)
            test_size    = 1 / (i + 1)
            
            task_train   = self.model_training(sample_size = sample_size, test_size = test_size)
            task_eval    = self.model_evaluate(task_id = task_train.task_id)

            yield task_train
            yield task_eval

            log.append({'i': i,
                        'train_path': task_train.output().path, 
                        'eval_path': task_eval.output_path, 
                        'sample_size': sample_size,
                        'test_size': test_size})

            if i > 2:
                break

        df = pd.DataFrame(log)
        df.to_csv(self.output().path)

        #ContextualBanDitsTraining_selu____512_54d17b4f72
        #output/models/ContextualBanditsTraining/results/ContextualBanditsTraining_selu____512_54d17b4f72
        #output/evaluation/EvaluateIfoodFullContentModel/results/ContextualBanditsTraining_selu____512_54d17b4f72_0e0e0de02a