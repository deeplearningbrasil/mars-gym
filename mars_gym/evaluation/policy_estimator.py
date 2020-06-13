from copy import deepcopy
from typing import List

import luigi
import numpy as np
from torch import nn as nn

from mars_gym.model.policy_estimator import PolicyEstimator
from mars_gym.meta_config import ProjectConfig, IOType
from mars_gym.simulation.base import BaseTorchModelTraining, TORCH_WEIGHT_INIT


class PolicyEstimatorTraining(BaseTorchModelTraining):
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

    def create_module(self) -> nn.Module:
        input_columns = self.project_config.input_columns
        num_elements_per_embeddings = [
            np.max(self.train_data_frame[input_column.name].values.tolist()) + 1
            for input_column in input_columns
            if input_column.type in (IOType.INDEXABLE, IOType.INDEXABLE_ARRAY)
        ]
        return PolicyEstimator(
            n_items=self.n_items,
            embedding_dim=self.embedding_dim,
            input_columns=input_columns,
            num_elements_per_embeddings=num_elements_per_embeddings,
            layers=self.layers,
            sample_batch=self.get_sample_batch(),
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
        )
