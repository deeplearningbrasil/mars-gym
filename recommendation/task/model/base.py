import abc
import gc
import importlib
import json
import logging
import os
import random
import shutil
from contextlib import redirect_stdout
from copy import deepcopy
from typing import Type, Dict, List, Optional, Tuple, Union, Any

import luigi
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchbearer
from torch.nn.init import xavier_normal
from torch.optim import Adam, RMSprop, SGD
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adamax import Adamax
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataset import Dataset
from torchbearer import Trial
from torchbearer.callbacks import GradientNormClipping
from torchbearer.callbacks.checkpointers import ModelCheckpoint
from torchbearer.callbacks.csv_logger import CSVLogger
from torchbearer.callbacks.early_stopping import EarlyStopping
from torchbearer.callbacks.tensor_board import TensorBoard
from tqdm import tqdm

from recommendation.data import preprocess_interactions_data_frame, preprocess_metadata_data_frame, \
    literal_eval_array_columns, InteractionsDataset
from recommendation.files import get_params_path, get_weights_path, get_interaction_dir, get_params, get_history_path, \
    get_tensorboard_logdir, get_task_dir, get_test_set_predictions_path
from recommendation.gym.envs.recsys import ITEM_METADATA_KEY
from recommendation.loss import ImplicitFeedbackBCELoss, CounterfactualRiskMinimization, FocalLoss
from recommendation.model.agent import BanditAgent
from recommendation.model.bandit import BANDIT_POLICIES
from recommendation.plot import plot_history
from recommendation.summary import summary
from recommendation.task.config import PROJECTS, ProjectConfig
from recommendation.task.cuda import CudaRepository
from recommendation.task.meta_config import Column, IOType
from recommendation.torch import NoAutoCollationDataLoader, RAdam, FasterBatchSampler
from recommendation.utils import lecun_normal_init, he_init

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TORCH_OPTIMIZERS = dict(adam=Adam, rmsprop=RMSprop, sgd=SGD, adadelta=Adadelta, adagrad=Adagrad, adamax=Adamax,
                        radam=RAdam)
TORCH_LOSS_FUNCTIONS = dict(mse=nn.MSELoss, nll=nn.NLLLoss, bce=nn.BCELoss, mlm=nn.MultiLabelMarginLoss,
                            implicit_feedback_bce=ImplicitFeedbackBCELoss, crm=CounterfactualRiskMinimization,
                            focal_loss=FocalLoss)
TORCH_ACTIVATION_FUNCTIONS = dict(relu=F.relu, selu=F.selu, tanh=F.tanh, sigmoid=F.sigmoid, linear=F.linear)
TORCH_WEIGHT_INIT = dict(lecun_normal=lecun_normal_init, he=he_init, xavier_normal=xavier_normal)
TORCH_DROPOUT_MODULES = dict(dropout=nn.Dropout, alpha=nn.AlphaDropout)

SEED = 42

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseModelTraining(luigi.Task):
    __metaclass__ = abc.ABCMeta

    project: str = luigi.ChoiceParameter(choices=PROJECTS.keys())

    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)
    session_test_size: float = luigi.FloatParameter(default=0.10)
    test_size: float = luigi.FloatParameter(default=0.0)
    dataset_split_method: str = luigi.ChoiceParameter(choices=["holdout", "column", "time", "k_fold"], default="time")
    test_split_type: str = luigi.ChoiceParameter(choices=["random", "time"], default="random")
    val_size: float = luigi.FloatParameter(default=0.2)
    n_splits: int = luigi.IntParameter(default=5)
    split_index: int = luigi.IntParameter(default=0)
    data_frames_preparation_extra_params: dict = luigi.DictParameter(default={})
    sampling_strategy: str = luigi.ChoiceParameter(choices=["oversample", "undersample", "none"], default="none")
    balance_fields: List[str] = luigi.ListParameter(default=[])
    sampling_proportions: Dict[str, Dict[str, float]] = luigi.DictParameter(default={})
    use_sampling_in_validation: bool = luigi.BoolParameter(default=False)
    eq_filters: Dict[str, any] = luigi.DictParameter(default={})
    neq_filters: Dict[str, any] = luigi.DictParameter(default={})
    isin_filters: Dict[str, any] = luigi.DictParameter(default={})
    seed: int = luigi.IntParameter(default=SEED)
    observation: str = luigi.Parameter(default="")

    negative_proportion: int = luigi.FloatParameter(0.0)

    @property
    def cache_attrs(self):
        return ['_test_dataset', '_val_dataset', '_train_dataset',
                '_test_data_frame', '_val_data_frame', '_train_data_frame', '_metadata_data_frame']

    def requires(self):
        return self.prepare_data_frames

    @property
    def prepare_data_frames(self):
        return self.project_config.prepare_data_frames_task(session_test_size=self.session_test_size,
                                                            sample_size=self.sample_size,
                                                            minimum_interactions=self.minimum_interactions,
                                                            test_size=self.test_size,
                                                            dataset_split_method=self.dataset_split_method,
                                                            test_split_type=self.test_split_type,
                                                            val_size=self.val_size,
                                                            n_splits=self.n_splits,
                                                            split_index=self.split_index,
                                                            sampling_strategy=self.sampling_strategy,
                                                            sampling_proportions=self.sampling_proportions,
                                                            balance_fields=self.balance_fields or self.project_config.default_balance_fields,
                                                            use_sampling_in_validation=self.use_sampling_in_validation,
                                                            eq_filters=self.eq_filters,
                                                            neq_filters=self.neq_filters,
                                                            isin_filters=self.isin_filters,
                                                            seed=self.seed,
                                                            **self.data_frames_preparation_extra_params)

    def output(self):
        return luigi.LocalTarget(get_task_dir(self.__class__, self.task_id))

    @property
    def project_config(self) -> ProjectConfig:
        if not hasattr(self, "_project_config"):
            self._project_config = deepcopy(PROJECTS[self.project])
            if self.loss_function == "crm" \
                    and self.project_config.propensity_score_column_name not in self.project_config.auxiliar_output_columns:
                self._project_config.auxiliar_output_columns = [
                    *self._project_config.auxiliar_output_columns,
                    Column(self.project_config.propensity_score_column_name, IOType.NUMBER)]
        return self._project_config

    def _save_params(self):
        with open(get_params_path(self.output().path), "w") as params_file:
            json.dump(self.param_kwargs, params_file, default=lambda o: dict(o), indent=4)

    @property
    def train_data_frame_path(self) -> str:
        return self.input()[0].path

    @property
    def val_data_frame_path(self) -> str:
        return self.input()[1].path

    @property
    def test_data_frame_path(self) -> str:
        return self.input()[2].path

    @property
    def metadata_data_frame_path(self) -> Optional[str]:
        if len(self.input()) > 3:
            return self.input()[3].path
        else:
            return None

    @property
    def metadata_data_frame(self) -> Optional[pd.DataFrame]:
        if not hasattr(self, "_metadata_data_frame"):
            self._metadata_data_frame = pd.read_csv(self.metadata_data_frame_path)\
                if self.metadata_data_frame_path else None
            if self._metadata_data_frame is not None:
                literal_eval_array_columns(self._metadata_data_frame, self.project_config.metadata_columns)
        return self._metadata_data_frame

    @property
    def embeddings_for_metadata(self) -> Optional[Dict[str, np.ndarray]]:
        if not hasattr(self, "_embeddings_for_metadata"):
            self._embeddings_for_metadata = preprocess_metadata_data_frame(
                self.metadata_data_frame, self.project_config) if self.metadata_data_frame is not None else None
        return self._embeddings_for_metadata

    @property
    def train_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_train_data_frame"):
            self._train_data_frame = preprocess_interactions_data_frame(pd.read_csv(self.train_data_frame_path),
                                                                        self.project_config)
        return self._train_data_frame

    @property
    def val_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_val_data_frame"):
            self._val_data_frame = preprocess_interactions_data_frame(pd.read_csv(self.val_data_frame_path),
                                                                      self.project_config)
        return self._val_data_frame

    @property
    def test_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_test_data_frame"):
            self._test_data_frame = preprocess_interactions_data_frame(pd.read_csv(self.test_data_frame_path),
                                                                       self.project_config)
        return self._test_data_frame

    @property
    def train_dataset(self) -> Dataset:
        if not hasattr(self, "_train_dataset"):
            self._train_dataset = self.project_config.dataset_class(
                data_frame=self.train_data_frame, embeddings_for_metadata=self.embeddings_for_metadata,
                project_config=self.project_config, negative_proportion=self.negative_proportion)
        return self._train_dataset

    @property
    def val_dataset(self) -> Dataset:
        if not hasattr(self, "_val_dataset"):
            self._val_dataset = self.project_config.dataset_class(
                data_frame=self.val_data_frame, embeddings_for_metadata=self.embeddings_for_metadata,
                project_config=self.project_config, negative_proportion=self.negative_proportion)
        return self._val_dataset

    @property
    def test_dataset(self) -> Dataset:
        if not hasattr(self, "_test_dataset"):
            self._test_dataset = self.project_config.dataset_class(
                data_frame=self.test_data_frame, embeddings_for_metadata=self.embeddings_for_metadata,
                project_config=self.project_config, negative_proportion=0.0)
        return self._test_dataset

    @property
    def vocab_size(self):
        if not hasattr(self, "_vocab_size"):
            self._vocab_size = int(self.train_data_frame.iloc[0]["vocab_size"])
        return self._vocab_size

    @property
    def n_users(self) -> int:
        if not hasattr(self, "_n_users"):
            self._n_users = int(self.train_data_frame.iloc[0][self.project_config.n_users_column])
        return self._n_users

    @property
    def n_items(self) -> int:
        if not hasattr(self, "_n_items"):
            self._n_items = int(self.train_data_frame.iloc[0][self.project_config.n_items_column])
        return self._n_items

    @abc.abstractmethod
    def train(self):
        pass

    def cache_cleanup(self):
        for a in self.cache_attrs:
            if hasattr(self, a):
                delattr(self, a)

    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(self):
        self.seed_everything()

        os.makedirs(self.output().path, exist_ok=True)
        self._save_params()
        try:
            self.train()
        except Exception:
            shutil.rmtree(self.output().path)
            raise
        finally:
            gc.collect()
            if self.device == "cuda":
                CudaRepository.put_available_device(self.device_id)


class BaseTorchModelTraining(BaseModelTraining):
    __metaclass__ = abc.ABCMeta

    device: str = luigi.ChoiceParameter(choices=["cpu", "cuda"], default=DEFAULT_DEVICE)

    batch_size: int = luigi.IntParameter(default=500)
    epochs: int = luigi.IntParameter(default=100)
    optimizer: str = luigi.ChoiceParameter(choices=TORCH_OPTIMIZERS.keys(), default="adam")
    optimizer_params: dict = luigi.DictParameter(default={})
    learning_rate: float = luigi.FloatParameter(1e-3)
    loss_function: str = luigi.ChoiceParameter(choices=TORCH_LOSS_FUNCTIONS.keys(), default="mse")
    loss_function_params: dict = luigi.DictParameter(default={})
    gradient_norm_clipping: float = luigi.FloatParameter(default=0.0)
    gradient_norm_clipping_type: float = luigi.IntParameter(default=2)
    early_stopping_patience: int = luigi.IntParameter(default=5)
    early_stopping_min_delta: float = luigi.FloatParameter(default=1e-3)
    monitor_metric: str = luigi.Parameter(default="val_loss")
    monitor_mode: str = luigi.Parameter(default="min")
    generator_workers: int = luigi.IntParameter(default=0)
    pin_memory: bool = luigi.BoolParameter(default=False)

    metrics = luigi.ListParameter(default=["loss"])

    @property
    def resources(self):
        return {"cuda": 1} if self.device == "cuda" else {}

    @property
    def device_id(self):
        if not hasattr(self, "_device_id"):
            if self.device == "cuda":
                self._device_id = CudaRepository.get_avaliable_device()
            else:
                self._device_id = None
        return self._device_id

    @abc.abstractmethod
    def create_module(self) -> nn.Module:
        pass

    def train(self):
        if self.device == "cuda":
            torch.cuda.set_device(self.device_id)

        train_loader    = self.get_train_generator()
        val_loader      = self.get_val_generator()
        module          = self.create_module()

        summary_path = os.path.join(self.output().path, "summary.txt")
        with open(summary_path, "w") as summary_file:
            with redirect_stdout(summary_file):
                sample_input = self.get_sample_batch()
                summary(module, sample_input)
            summary(module, sample_input)

        trial = self.create_trial(module)

        try:
            trial.with_generators(train_generator=train_loader, val_generator=val_loader).run(epochs=self.epochs)
        except KeyboardInterrupt:
            print("Finishing the training at the request of the user...")

        history_df = pd.read_csv(get_history_path(self.output().path))

        plot_history(history_df).savefig(os.path.join(self.output().path, "history.jpg"))

        self.after_fit()
        self.evaluate()
        self.cache_cleanup()

    def get_sample_batch(self):
        return default_convert(self.train_dataset[0][0])

    def after_fit(self):
        pass

    def evaluate(self):
        module      = self.get_trained_module()
        val_loader  = self.get_val_generator()

        print("================== Evaluate ========================")
        trial = Trial(module, self._get_optimizer(module), self._get_loss_function(), callbacks=[],
                      metrics=self.metrics).to(self.torch_device)\
                    .with_generators(val_generator=val_loader).eval()

        print(json.dumps((trial.evaluate(data_key=torchbearer.VALIDATION_DATA)), indent = 4))


    def create_trial(self, module: nn.Module) -> Trial:
        loss_function = self._get_loss_function()
        trial = Trial(module, self._get_optimizer(module), loss_function, callbacks=self._get_callbacks(),
                      metrics=self.metrics).to(self.torch_device)
        if hasattr(loss_function, "torchbearer_state"):
            loss_function.torchbearer_state = trial.state
        return trial

    def _get_loss_function(self):
        return TORCH_LOSS_FUNCTIONS[self.loss_function](**self.loss_function_params)

    def _get_optimizer(self, module) -> Optimizer:
        return TORCH_OPTIMIZERS[self.optimizer](module.parameters(), lr=self.learning_rate,
                                                **self.optimizer_params)

    def _get_callbacks(self):
        callbacks = [
            *self._get_extra_callbacks(),
            ModelCheckpoint(get_weights_path(self.output().path), save_best_only=True, monitor=self.monitor_metric,
                            mode=self.monitor_mode),
            EarlyStopping(patience=self.early_stopping_patience, min_delta=self.early_stopping_min_delta,
                          monitor=self.monitor_metric, mode=self.monitor_mode),
            CSVLogger(get_history_path(self.output().path)),
            TensorBoard(get_tensorboard_logdir(self.task_id), write_graph=False),
        ]
        if self.gradient_norm_clipping:
            callbacks.append(GradientNormClipping(self.gradient_norm_clipping, self.gradient_norm_clipping_type))
        return callbacks

    def _get_extra_callbacks(self):
        return []

    def get_trained_module(self) -> nn.Module:
        module = self.create_module().to(self.torch_device)
        state_dict = torch.load(get_weights_path(self.output().path), map_location=self.torch_device)
        module.load_state_dict(state_dict["model"])
        module.eval()
        return module

    @property
    def torch_device(self) -> torch.device:
        if not hasattr(self, "_torch_device"):
            if self.device == "cuda":
                self._torch_device = torch.device(f"cuda:{self.device_id}")
            else:
                self._torch_device = torch.device("cpu")
        return self._torch_device

    def get_train_generator(self) -> DataLoader:
        batch_sampler = FasterBatchSampler(self.train_dataset, self.batch_size, shuffle=True)
        return NoAutoCollationDataLoader(self.train_dataset, batch_sampler=batch_sampler,
                                         num_workers=self.generator_workers,
                                         pin_memory=self.pin_memory if self.device == "cuda" else False)

    def get_val_generator(self) -> Optional[DataLoader]:
        if len(self.val_data_frame) == 0:
            return None
        batch_sampler = FasterBatchSampler(self.val_dataset, self.batch_size, shuffle=False)
        return NoAutoCollationDataLoader(self.val_dataset, batch_sampler=batch_sampler,
                                         num_workers=self.generator_workers,
                                         pin_memory=self.pin_memory if self.device == "cuda" else False)

    def get_test_generator(self) -> DataLoader:
        batch_sampler = FasterBatchSampler(self.test_dataset, self.batch_size, shuffle=False)
        return NoAutoCollationDataLoader(self.test_dataset, batch_sampler=batch_sampler,
                                         num_workers=self.generator_workers,
                                         pin_memory=True if self.device == "cuda" else False)


class BaseTorchModelWithAgentTraining(BaseTorchModelTraining):
    bandit_policy: str = luigi.ChoiceParameter(choices=BANDIT_POLICIES.keys(), default="model")
    bandit_policy_params: Dict[str, Any] = luigi.DictParameter(default={})

    def create_agent(self) -> BanditAgent:
        bandit = BANDIT_POLICIES[self.bandit_policy](reward_model=self.create_module(), **self.bandit_policy_params)
        return BanditAgent(bandit)

    @property
    def unique_items(self) -> List[int]:
        if not hasattr(self, "_unique_items"):
            self._unique_items = self.interactions_data_frame[self.project_config.item_column.name].unique()
        return self._unique_items

    @property
    def obs_columns(self) -> List[str]:
        if not hasattr(self, "_obs_columns"):
            self._obs_columns = [self.project_config.user_column.name] + [
                column.name for column in self.project_config.other_input_columns]
        return self._obs_columns

    def _get_arm_indices(self, ob: dict) -> Union[List[int], np.ndarray]:
        if self.project_config.available_arms_column_name:
            arm_indices = np.flatnonzero(ob[self.project_config.available_arms_column_name])
        else:
            arm_indices = self.unique_items

        return arm_indices

    def _get_arm_scores(self, agent: BanditAgent, ob_dataset: Dataset) -> List[float]:
        batch_sampler = FasterBatchSampler(ob_dataset, self.batch_size, shuffle=False)
        generator     = NoAutoCollationDataLoader(ob_dataset,
                            batch_sampler=batch_sampler, num_workers=self.generator_workers,
                            pin_memory=self.pin_memory if self.device == "cuda" else False)

        trial = Trial(agent.bandit.reward_model, criterion=lambda *args: torch.zeros(1, device=self.torch_device,
                                                                                     requires_grad=True)) \
            .with_test_generator(generator).to(self.torch_device).eval()

        with torch.no_grad():
            model_output: Union[torch.Tensor, Tuple[torch.Tensor]] = trial.predict(verbose=0)

        scores_tensor: torch.Tensor = model_output if isinstance(model_output, torch.Tensor) else model_output[0][0]
        scores: List[float] = scores_tensor.cpu().numpy().reshape(-1).tolist()

        return scores

    def _create_ob_dataset(self, ob: dict, arm_indices: List[int]) -> Dataset:
        data = [{**ob, self.project_config.item_column.name: arm_index} for arm_index in arm_indices]
        ob_df = pd.DataFrame(
            columns=self.obs_columns + [self.project_config.item_column.name],
            data=data)

        ob_df = self._fill_hist_columns(ob_df)

        if self.project_config.output_column.name not in ob_df.columns:
            ob_df[self.project_config.output_column.name] = 1
        for auxiliar_output_column in self.project_config.auxiliar_output_columns:
            if auxiliar_output_column.name not in ob_df.columns:
                ob_df[auxiliar_output_column.name] = 0

        dataset = InteractionsDataset(ob_df, ob[ITEM_METADATA_KEY], self.project_config)

        return dataset

    def _fill_hist_columns(self, ob_df: pd.DataFrame) -> pd.DataFrame:
        if self.project_config.hist_view_column_name not in ob_df:
            ob_df[self.project_config.hist_view_column_name] = 0
        if self.project_config.hist_output_column_name not in ob_df:
            ob_df[self.project_config.hist_output_column_name] = 0
        return ob_df

    def _prepare_for_agent(self, agent: BanditAgent, ob: dict) -> Tuple[np.ndarray, List[int], List[float]]:
        arm_indices = self._get_arm_indices(ob)
        ob_dataset = self._create_ob_dataset(ob, arm_indices)
        arm_contexts = ob_dataset[:len(ob_dataset)][0]
        arm_scores = self._get_arm_scores(agent, ob_dataset) \
            if agent.bandit.reward_model else agent.bandit.calculate_scores(arm_contexts)
        return arm_contexts, arm_indices, arm_scores

    def _act(self, agent: BanditAgent, ob: dict) -> int:
        arm_contexts, arm_indices, arm_scores = self._prepare_for_agent(agent, ob)

        return agent.act(arm_indices, arm_contexts, arm_scores)

    def _rank(self, agent: BanditAgent,  ob: dict) -> Tuple[List[int], List[float], List[float]]:
        arm_contexts, arm_indices, arm_scores = self._prepare_for_agent(agent, ob)
        sorted_actions, proba_actions = agent.rank(arm_indices, arm_contexts, arm_scores)

        return sorted_actions, proba_actions, arm_scores

    def clean(self):
        if hasattr(self, "_train_dataset"):
            del self._train_dataset

        if hasattr(self, "_test_dataset"):
            del self._test_dataset

        if hasattr(self, "_train_data_frame"):
            del self._train_data_frame

        gc.collect()

    def _save_test_set_predictions(self, agent: BanditAgent) -> None:
        print("Saving test set predictions...")
        sorted_actions_list = []
        prob_actions_list = []
        action_scores_list = []
        obs = self.test_data_frame.to_dict('records')
        self.clean()
        # from IPython import embed; embed()
        for ob in tqdm(obs, total=len(obs)):
            if self.embeddings_for_metadata is not None:
                ob[ITEM_METADATA_KEY] = self.embeddings_for_metadata

            if self.project_config.available_arms_column_name:
                items = np.zeros(np.max(ob[self.project_config.available_arms_column_name]) + 1)
                items[ob[self.project_config.available_arms_column_name]] = 1
                ob[self.project_config.available_arms_column_name] = items

            sorted_actions, prob_actions, action_scores = self._rank(agent, ob)
            sorted_actions_list.append(sorted_actions)
            prob_actions_list.append(prob_actions)

            if action_scores:
                action_scores_list.append(list(reversed(sorted(action_scores))))

            del ob
            del items

        self.test_data_frame["sorted_actions"] = sorted_actions_list
        self.test_data_frame["prob_actions"] = prob_actions_list

        if action_scores_list:
            self.test_data_frame["action_scores"] = action_scores_list

        self.test_data_frame.to_csv(get_test_set_predictions_path(self.output().path), index=False)

    def after_fit(self):
        if self.test_size > 0:
            self._save_test_set_predictions(self.create_agent())


class BaseEvaluationTask(luigi.Task, metaclass=abc.ABCMeta):
    model_module: str = luigi.Parameter(default="recommendation.task.model.interaction")
    model_cls: str = luigi.Parameter(default="InteractionTraining")
    model_task_id: str = luigi.Parameter()
    no_offpolicy_eval: bool = luigi.BoolParameter(default=False)
    task_hash: str = luigi.Parameter(default='none')

    @property
    def cache_attr(self):
        return ['']

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
    def n_items(self):
        return self.model_training.n_items

    def output(self):
        return luigi.LocalTarget(os.path.join("output", "evaluation", self.__class__.__name__, "results",
                                              self.task_name))

    def cache_cleanup(self):
        for a in self.cache_attrs:
            if hasattr(self, a):
                delattr(self, a)

    def _save_params(self):
        with open(get_params_path(self.output().path), "w") as params_file:
            json.dump(self.param_kwargs, params_file, default=lambda o: dict(o), indent=4)

def load_torch_model_training_from_task_dir(model_cls: Type[BaseTorchModelTraining],
                                            task_dir: str) -> BaseTorchModelTraining:
    return model_cls(**get_params(task_dir))


def load_torch_model_training_from_task_id(model_cls: Type[BaseTorchModelTraining],
                                           task_id: str) -> BaseTorchModelTraining:
    task_dir = get_task_dir(model_cls, task_id)
    if not os.path.exists(task_dir):
        task_dir = get_interaction_dir(model_cls, task_id)

    return load_torch_model_training_from_task_dir(model_cls, task_dir)
