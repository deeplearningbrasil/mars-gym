
import torch.nn as nn

import luigi

from recommendation.model.matrix_factorization import MatrixFactorization, BiasedMatrixFactorization
from recommendation.task.model.base import BaseTorchModelTraining, TORCH_WEIGHT_INIT


class MatrixFactorizationTraining(BaseTorchModelTraining):
    biased: bool = luigi.BoolParameter(default=False)
    n_factors: int = luigi.IntParameter(default=20)
    weight_init: str = luigi.ChoiceParameter(choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal")

    def create_module(self) -> nn.Module:
        model_cls = BiasedMatrixFactorization if self.biased else MatrixFactorization
        return model_cls(
            n_users=self.project_config.n_users,
            n_items=self.project_config.n_items,
            n_factors=self.n_factors,
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
        )
