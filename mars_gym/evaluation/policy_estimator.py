from copy import deepcopy
from typing import List, Type

import luigi
import numpy as np
from torch import nn as nn

from mars_gym.model.abstract import RecommenderModule
from mars_gym.model.policy_estimator import PolicyEstimator
from mars_gym.meta_config import ProjectConfig, IOType
from mars_gym.simulation.training import TorchModelTraining, TORCH_WEIGHT_INIT


class PolicyEstimatorTraining(TorchModelTraining):
    recommender_module_class: str = None
    recommender_extra_params: dict = None

    val_size = 0.0
    test_size = 0.0
    monitor_metric = "loss"
    loss_function = "nll"

    layers: List[int] = luigi.ListParameter(default=[])  # 1000
    weight_init: str = luigi.ChoiceParameter(
        choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal"
    )

    embedding_dim: int = luigi.IntParameter(default=50)
    epochs = luigi.IntParameter(default=500)
    early_stopping_patience: int = luigi.IntParameter(default=20)
    metrics = luigi.ListParameter(default=["loss", "acc"])

    @property
    def project_config(self) -> ProjectConfig:
        if not hasattr(self, "_project_config"):
            project_config = deepcopy(super().project_config)
            project_config.output_column = project_config.item_column
            project_config.item_is_input = False
            self._project_config = project_config
        return self._project_config

    @property
    def module_class(self) -> Type[RecommenderModule]:
        return PolicyEstimator

    def create_module(self) -> nn.Module:
        return PolicyEstimator(
            index_mapping=self.index_mapping,
            project_config=self.project_config,
            embedding_dim=self.embedding_dim,
            layers=self.layers,
            sample_batch=self.get_sample_batch(),
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
        )
