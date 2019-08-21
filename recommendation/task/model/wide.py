from typing import List

import torch.nn as nn

import luigi

from recommendation.model.wide import WideCTR
from recommendation.task.meta_config import RecommenderType
from recommendation.task.model.base import BaseTorchModelTraining, TORCH_ACTIVATION_FUNCTIONS, TORCH_WEIGHT_INIT, \
    TORCH_DROPOUT_MODULES
from recommendation.torch import MaskedZeroesLoss


class WideTraining(BaseTorchModelTraining):
    input_dim: int = luigi.IntParameter(default=100)
    dropout_prob: float = luigi.FloatParameter(default=None)
    activation_function: str = luigi.ChoiceParameter(choices=TORCH_ACTIVATION_FUNCTIONS.keys(), default="selu")
    weight_init: str = luigi.ChoiceParameter(choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal")
    dropout_module: str = luigi.ChoiceParameter(choices=TORCH_DROPOUT_MODULES.keys(), default="alpha")


    def create_module(self) -> nn.Module:
        return WideCTR(self.input_dim, 
                        dropout_prob=self.dropout_prob,
                        activation_function=TORCH_ACTIVATION_FUNCTIONS[self.activation_function],
                        weight_init=TORCH_WEIGHT_INIT[self.weight_init],
                        dropout_module=TORCH_DROPOUT_MODULES[self.dropout_module])
