import os
from typing import List

import numpy as np
import torch.nn as nn
import luigi

from recommendation.model.matrix_factorization import MatrixFactorization, BiasedMatrixFactorization, \
    DeepMatrixFactorization, MatrixFactorizationWithShift, MatrixFactorizationWithShiftTime
from recommendation.task.model.base import BaseTorchModelTraining, TORCH_WEIGHT_INIT, TORCH_DROPOUT_MODULES, \
    TORCH_ACTIVATION_FUNCTIONS, TORCH_LOSS_FUNCTIONS
from recommendation.task.model.embedding import UserAndItemEmbeddingTraining


class MatrixFactorizationTraining(UserAndItemEmbeddingTraining):
    biased: bool = luigi.BoolParameter(default=False)
    binary: bool = luigi.BoolParameter(default=False)

    def create_module(self) -> nn.Module:
        model_cls = BiasedMatrixFactorization if self.biased else MatrixFactorization
        return model_cls(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            binary=self.binary,
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
        )


class MatrixFactorizationWithShiftTraining(UserAndItemEmbeddingTraining):
    loss_function: str = luigi.ChoiceParameter(choices=TORCH_LOSS_FUNCTIONS.keys(), default="bce")

    user_shift_combination: str = luigi.ChoiceParameter(choices=["sum", "multiply"], default="sum")

    def create_module(self) -> nn.Module:
        return MatrixFactorizationWithShift(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
            user_shift_combination=self.user_shift_combination,
        )

class MatrixFactorizationWithShiftTimeTraining(UserAndItemEmbeddingTraining):
    loss_function: str = luigi.ChoiceParameter(choices=TORCH_LOSS_FUNCTIONS.keys(), default="bce")

    user_shift_combination: str = luigi.ChoiceParameter(choices=["sum", "multiply"], default="sum")

    def create_module(self) -> nn.Module:
        return MatrixFactorizationWithShiftTime(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
            user_shift_combination=self.user_shift_combination,
        )

class DeepMatrixFactorizationTraining(UserAndItemEmbeddingTraining):
    dense_layers: List[int] = luigi.ListParameter(default=[64, 32])
    dropout_between_layers_prob: float = luigi.FloatParameter(default=None)
    activation_function: str = luigi.ChoiceParameter(choices=TORCH_ACTIVATION_FUNCTIONS.keys(), default="selu")
    dropout_module: str = luigi.ChoiceParameter(choices=TORCH_DROPOUT_MODULES.keys(), default="alpha")
    bn_between_layers: bool = luigi.BoolParameter(default=False)
    binary: bool = luigi.BoolParameter(default=False)

    def create_module(self) -> nn.Module:
        return DeepMatrixFactorization(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            binary=self.binary,
            dense_layers=self.dense_layers,
            dropout_between_layers_prob=self.dropout_between_layers_prob,
            bn_between_layers=self.bn_between_layers,
            activation_function=TORCH_ACTIVATION_FUNCTIONS[self.activation_function],
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
            dropout_module=TORCH_DROPOUT_MODULES[self.dropout_module],
        )


