from typing import List

import luigi
import torch.nn as nn

from recommendation.model.trivago_bandits import LogisticRegression
from recommendation.task.model.base import TORCH_ACTIVATION_FUNCTIONS, TORCH_DROPOUT_MODULES
from recommendation.task.model.base import TORCH_WEIGHT_INIT
from recommendation.task.model.interaction import InteractionTraining
from recommendation.task.model.base import BaseTorchModelTraining

class TrivagoBanditsTraining(BaseTorchModelTraining):
    loss_function: str = luigi.ChoiceParameter(choices=["crm", "bce"], default="bce")
    n_factors: int = luigi.IntParameter(default=128)
    weight_init: str = luigi.ChoiceParameter(choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal")
    dropout_module: str = luigi.ChoiceParameter(choices=TORCH_DROPOUT_MODULES.keys(), default="alpha")
    activation_function: str = luigi.ChoiceParameter(choices=TORCH_ACTIVATION_FUNCTIONS.keys(), default="selu")
    word_embeddings_size: int = luigi.IntParameter(default=128)

    def create_module(self) -> nn.Module:
        return LogisticRegression(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors
        )
