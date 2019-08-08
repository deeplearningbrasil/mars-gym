import os
from typing import List

import numpy as np
import torch.nn as nn
import luigi

from recommendation.model.matrix_factorization import MatrixFactorization, BiasedMatrixFactorization, \
    DeepMatrixFactorization
from recommendation.task.model.base import BaseTorchModelTraining, TORCH_WEIGHT_INIT, TORCH_DROPOUT_MODULES, \
    TORCH_ACTIVATION_FUNCTIONS


class MatrixFactorizationTraining(BaseTorchModelTraining):
    biased: bool = luigi.BoolParameter(default=False)
    n_factors: int = luigi.IntParameter(default=20)
    binary: bool = luigi.BoolParameter(default=False)
    weight_init: str = luigi.ChoiceParameter(choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal")

    def create_module(self) -> nn.Module:
        model_cls = BiasedMatrixFactorization if self.biased else MatrixFactorization
        return model_cls(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            binary=self.binary,
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
        )


class DeepMatrixFactorizationTraining(BaseTorchModelTraining):
    n_factors: int = luigi.IntParameter(default=20)

    dense_layers: List[int] = luigi.ListParameter(default=[64, 32])
    dropout_between_layers_prob: float = luigi.FloatParameter(default=None)
    activation_function: str = luigi.ChoiceParameter(choices=TORCH_ACTIVATION_FUNCTIONS.keys(), default="selu")
    weight_init: str = luigi.ChoiceParameter(choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal")
    dropout_module: str = luigi.ChoiceParameter(choices=TORCH_DROPOUT_MODULES.keys(), default="alpha")
    bn_between_layers: bool = luigi.BoolParameter(default=False)

    def create_module(self) -> nn.Module:
        return DeepMatrixFactorization(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            dense_layers=self.dense_layers,
            dropout_between_layers_prob=self.dropout_between_layers_prob,
            bn_between_layers=self.bn_between_layers,
            activation_function=TORCH_ACTIVATION_FUNCTIONS[self.activation_function],
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
            dropout_module=TORCH_DROPOUT_MODULES[self.dropout_module]
        )

    def after_fit(self):
        module: MatrixFactorization = self.get_trained_module()
        item_embeddings: np.ndarray = module.item_embeddings.weight.data.cpu().numpy()
        np.savetxt(os.path.join(self.output().path, "item_embeddings.tsv"), item_embeddings, delimiter="\t")
