import abc
import functools
import json
import logging
import multiprocessing
import os
import shutil
from contextlib import redirect_stdout
from typing import Type, Dict, List

import luigi
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal
from torch.optim import Adam, RMSprop, SGD
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adamax import Adamax
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchbearer import Trial
from torchbearer.callbacks.checkpointers import ModelCheckpoint
from torchbearer.callbacks.csv_logger import CSVLogger
from torchbearer.callbacks.early_stopping import EarlyStopping
from torchbearer.callbacks.tensor_board import TensorBoard
from torchbearer.callbacks.torch_scheduler import CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau, StepLR
import mlflow

from recommendation.data import SupportBasedCorruptionTransformation, \
    MaskingNoiseCorruptionTransformation, SaltAndPepperNoiseCorruptionTransformation
from recommendation.files import get_params_path, get_weights_path, get_params, get_history_path, \
    get_tensorboard_logdir, get_task_dir
from recommendation.plot import plot_history, plot_loss_per_lr, plot_loss_derivatives_per_lr
from recommendation.summary import summary
from recommendation.task.config import PROJECTS, IOType
from recommendation.task.cuda import CudaRepository
from recommendation.torch import MLFlowLogger, CosineAnnealingWithRestartsLR, CyclicLR, LearningRateFinder, collate_fn, \
    FocalLoss
from recommendation.utils import lecun_normal_init, he_init

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TORCH_DATA_TRANSFORMATIONS = dict(support_based=SupportBasedCorruptionTransformation,
                                  masking_noise=MaskingNoiseCorruptionTransformation,
                                  salt_and_pepper_noise=SaltAndPepperNoiseCorruptionTransformation,
                                  none=None)
TORCH_OPTIMIZERS = dict(adam=Adam, rmsprop=RMSprop, sgd=SGD, adadelta=Adadelta, adagrad=Adagrad, adamax=Adamax)
TORCH_LOSS_FUNCTIONS = dict(mse=nn.MSELoss, nll=nn.NLLLoss, bce=nn.BCELoss, mlm=nn.MultiLabelMarginLoss,
                            focal=FocalLoss)
TORCH_ACTIVATION_FUNCTIONS = dict(relu=F.relu, selu=F.selu, tanh=F.tanh, sigmoid=F.sigmoid, linear=F.linear)
TORCH_WEIGHT_INIT = dict(lecun_normal=lecun_normal_init, he=he_init, xavier_normal=xavier_normal)
TORCH_DROPOUT_MODULES = dict(dropout=nn.Dropout, alpha=nn.AlphaDropout)
TORCH_LR_SCHEDULERS = dict(step=StepLR, exponential=ExponentialLR, cosine_annealing=CosineAnnealingLR,
                           cosine_annealing_with_restarts=CosineAnnealingWithRestartsLR, cyclic=CyclicLR,
                           reduce_on_plateau=ReduceLROnPlateau)

SEED = 42

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseModelTraining(luigi.Task):
    __metaclass__ = abc.ABCMeta

    project: str = luigi.ChoiceParameter(choices=PROJECTS.keys())

    data_transformation: str = luigi.ChoiceParameter(choices=TORCH_DATA_TRANSFORMATIONS.keys(), default="none")
    data_transformation_params: dict = luigi.DictParameter(default={})
    test_size: float = luigi.FloatParameter(default=0.2)
    dataset_split_method: str = luigi.ChoiceParameter(choices=["holdout", "k_fold"], default="holdout")
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

    def requires(self):
        return self.project_config.prepare_data_frames_task(test_size=self.test_size,
                                                            dataset_split_method=self.dataset_split_method,
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
    def project_config(self):
        return PROJECTS[self.project]

    @property
    def target_column(self) -> str:
        return self.target_col or self.project_config.default_target_col

    def _save_params(self):
        with open(get_params_path(self.output().path), "w") as params_file:
            json.dump(self.param_kwargs, params_file, default=lambda o: dict(o), indent=4)
        for key, value in self.param_kwargs.items():
            mlflow.log_param(key, value)

    @property
    def train_dataset(self) -> Dataset:
        if not hasattr(self, "_train_dataset"):
            train_df = pd.read_csv(self.input()[0].path)
            transformation = TORCH_DATA_TRANSFORMATIONS[self.data_transformation](**self.data_transformation_params) \
                if self.data_transformation != "none" else None
            self._train_dataset = self.project_config.dataset_class(train_df, self.project_config,
                                                                    transformation=transformation)
        return self._train_dataset

    @property
    def val_dataset(self) -> Dataset:
        if not hasattr(self, "_val_dataset"):
            val_df = pd.read_csv(self.input()[1].path)
            self._val_dataset = self.project_config.dataset_class(val_df, self.project_config)
        return self._val_dataset

    @property
    def test_dataset(self) -> Dataset:
        if not hasattr(self, "_test_dataset"):
            test_df = pd.read_csv(self.input()[2].path)
            self._test_dataset = self.project_config.dataset_class(test_df, self.project_config)
        return self._test_dataset

    @property
    def n_users(self):
        if not hasattr(self, "_n_users"):
            train_df = pd.read_csv(self.input()[0].path, nrows=1)
            self._n_users = int(train_df.iloc[0][self.project_config.n_users_column])
        return self._n_users

    @property
    def n_items(self):
        if not hasattr(self, "_n_items"):
            train_df = pd.read_csv(self.input()[0].path, nrows=1)
            self._n_items = int(train_df.iloc[0][self.project_config.n_items_column])
        return self._n_items

    @abc.abstractmethod
    def train(self):
        pass

    def run(self):
        np.random.seed(self.seed)

        mlflow.set_experiment(self.__class__.__name__)
        with mlflow.start_run(run_name=self.task_id):
            os.makedirs(self.output().path, exist_ok=True)
            self._save_params()
            try:
                self.train()
            except Exception:
                shutil.rmtree(self.output().path)
                raise
            finally:
                if self.device == "cuda":
                    CudaRepository.put_available_device(self.device_id)


class BaseTorchModelTraining(BaseModelTraining):
    __metaclass__ = abc.ABCMeta

    device: str = luigi.ChoiceParameter(choices=["cpu", "cuda"], default=DEFAULT_DEVICE)

    mode: str = luigi.ChoiceParameter(choices=["fit", "lr_find"], default="fit")
    lr_find_iterations: int = luigi.IntParameter(default=100)
    batch_size: int = luigi.IntParameter(default=500)
    epochs: int = luigi.IntParameter(default=100)
    optimizer: str = luigi.ChoiceParameter(choices=TORCH_OPTIMIZERS.keys(), default="adam")
    optimizer_params: dict = luigi.DictParameter(default={})
    learning_rate: float = luigi.FloatParameter(1e-3)
    lr_scheduler: str = luigi.ChoiceParameter(choices=TORCH_LR_SCHEDULERS.keys(), default=None)
    lr_scheduler_params: dict = luigi.DictParameter(default={})
    loss_function: str = luigi.ChoiceParameter(choices=TORCH_LOSS_FUNCTIONS.keys(), default="mse")
    loss_function_params: dict = luigi.DictParameter(default={})
    early_stopping_patience: int = luigi.IntParameter(default=10)
    early_stopping_min_delta: float = luigi.FloatParameter(default=1e-6)
    generator_workers: int = luigi.IntParameter(default=min(multiprocessing.cpu_count(), 20))
    pin_memory: bool = luigi.BoolParameter(default=False)

    metrics = luigi.ListParameter(default=["loss", "mse"])

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

        torch.manual_seed(self.seed)
        dict(fit=self.fit, lr_find=self.lr_find)[self.mode]()

    def fit(self):
        train_loader = self.get_train_generator()
        val_loader = self.get_val_generator()

        module = self.create_module()

        summary_path = os.path.join(self.output().path, "summary.txt")
        with open(summary_path, "w") as summary_file:
            with redirect_stdout(summary_file):
                sample_input = collate_fn([self.train_dataset[0][0]])
                summary(module, sample_input)
        mlflow.log_artifact(summary_path)

        trial = self.create_trial(module)

        try:
            trial.with_generators(train_generator=train_loader, val_generator=val_loader).run(epochs=self.epochs)
        except KeyboardInterrupt:
            print("Finishing the training at the request of the user...")

        history_df = pd.read_csv(get_history_path(self.output().path))

        plot_history(history_df).savefig(os.path.join(self.output().path, "history.jpg"))

        mlflow.log_artifact(get_weights_path(self.output().path))
        self.after_fit()

    def after_fit(self):
        pass

    def lr_find(self):
        train_loader = self.get_train_generator()

        self.learning_rate = 1e-6
        lr_finder = LearningRateFinder(min(len(train_loader), self.lr_find_iterations), self.learning_rate)

        module = self.create_module()

        trial = Trial(module, self._get_optimizer(module), self._get_loss_function(), callbacks=[lr_finder]) \
            .with_train_generator(train_loader) \
            .to(self.torch_device)

        trial.run(epochs=1)

        loss_per_lr_path = os.path.join(self.output().path, "loss_per_lr.jpg")
        loss_derivatives_per_lr_path = os.path.join(self.output().path, "loss_derivatives_per_lr.jpg")

        plot_loss_per_lr(lr_finder.learning_rates, lr_finder.loss_values) \
            .savefig(loss_per_lr_path)
        plot_loss_derivatives_per_lr(lr_finder.learning_rates, lr_finder.get_loss_derivatives(5)) \
            .savefig(loss_derivatives_per_lr_path)

        mlflow.log_artifact(loss_per_lr_path)
        mlflow.log_artifact(loss_derivatives_per_lr_path)

    def create_trial(self, module: nn.Module) -> Trial:
        return Trial(module, self._get_optimizer(module), self._get_loss_function(), callbacks=self._get_callbacks(),
                     metrics=self.metrics).to(self.torch_device)

    def _get_loss_function(self):
        return TORCH_LOSS_FUNCTIONS[self.loss_function](**self.loss_function_params)

    def _get_optimizer(self, module) -> Optimizer:
        return TORCH_OPTIMIZERS[self.optimizer](module.parameters(), lr=self.learning_rate,
                                                **self.optimizer_params)

    def _get_callbacks(self):
        callbacks = [
            *self._get_extra_callbacks(),
            ModelCheckpoint(get_weights_path(self.output().path), save_best_only=True),
            EarlyStopping(patience=self.early_stopping_patience, min_delta=self.early_stopping_min_delta,
                          verbose=True),
            CSVLogger(get_history_path(self.output().path)), MLFlowLogger(),
            TensorBoard(get_tensorboard_logdir(self.task_id)),
        ]
        if self.lr_scheduler:
            callbacks.append(TORCH_LR_SCHEDULERS[self.lr_scheduler](**self.lr_scheduler_params))
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
        fn = functools.partial(collate_fn, use_shared_memory=self.generator_workers > 0)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.generator_workers, collate_fn=fn,
                          pin_memory=self.pin_memory if self.device == "cuda" else False)

    def get_val_generator(self) -> DataLoader:
        fn = functools.partial(collate_fn, use_shared_memory=self.generator_workers > 0)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.generator_workers, collate_fn=fn,
                          pin_memory=self.pin_memory if self.device == "cuda" else False)

    def get_test_generator(self) -> DataLoader:
        fn = functools.partial(collate_fn, use_shared_memory=self.generator_workers > 0)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.generator_workers, collate_fn=fn,
                          pin_memory=True if self.device == "cuda" else False)


def load_torch_model_training_from_task_dir(model_cls: Type[BaseTorchModelTraining],
                                            task_dir: str) -> BaseTorchModelTraining:
    return model_cls(**get_params(task_dir))


def load_torch_model_training_from_task_id(model_cls: Type[BaseTorchModelTraining],
                                           task_id: str) -> BaseTorchModelTraining:
    task_dir = get_task_dir(model_cls, task_id)

    return load_torch_model_training_from_task_dir(model_cls, task_dir)
