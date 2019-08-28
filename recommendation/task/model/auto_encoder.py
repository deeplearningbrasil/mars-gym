from typing import List

import torch.nn as nn

import luigi

from recommendation.model.auto_encoder import UnconstrainedAutoEncoder, VariationalAutoEncoder
from recommendation.task.meta_config import RecommenderType
from recommendation.task.model.base import BaseTorchModelTraining, TORCH_ACTIVATION_FUNCTIONS, TORCH_WEIGHT_INIT, \
    TORCH_DROPOUT_MODULES
from recommendation.torch import MaskedZeroesLoss, SparseTensorLoss


class UnconstrainedAutoEncoderTraining(BaseTorchModelTraining):
    metrics = luigi.ListParameter(default=["loss", "precision", "recall"])

    encoder_layers: List[int] = luigi.ListParameter(default=[256, 128, 128, 64])
    decoder_layers: List[int] = luigi.ListParameter(default=[128, 128, 256])
    dropout_prob: float = luigi.FloatParameter(default=None)
    activation_function: str = luigi.ChoiceParameter(choices=TORCH_ACTIVATION_FUNCTIONS.keys(), default="selu")
    weight_init: str = luigi.ChoiceParameter(choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal")
    dropout_module: str = luigi.ChoiceParameter(choices=TORCH_DROPOUT_MODULES.keys(), default="alpha")
    masked_zeroes_loss: bool = luigi.BoolParameter(default=False)
    binary: bool = luigi.BoolParameter(default=False)

    def _get_loss_function(self):
        if self.masked_zeroes_loss:
            return MaskedZeroesLoss(super()._get_loss_function())
        else:
            return SparseTensorLoss(super()._get_loss_function())

    def create_module(self) -> nn.Module:
        dim = self.n_items \
            if self.project_config.recommender_type == RecommenderType.USER_BASED_COLLABORATIVE_FILTERING \
            else self.n_users
        return UnconstrainedAutoEncoder(dim, self.encoder_layers, self.decoder_layers, binary=self.binary,
                                        dropout_prob=self.dropout_prob,
                                        activation_function=TORCH_ACTIVATION_FUNCTIONS[self.activation_function],
                                        weight_init=TORCH_WEIGHT_INIT[self.weight_init],
                                        dropout_module=TORCH_DROPOUT_MODULES[self.dropout_module])


class VariationalAutoEncoderTraining(BaseTorchModelTraining):
    encoder_layers: List[int] = luigi.ListParameter(default=[256, 128, 128])
    decoder_layers: List[int] = luigi.ListParameter(default=[128, 128, 256])
    dropout_prob: float = luigi.FloatParameter(default=None)
    activation_function: str = luigi.ChoiceParameter(choices=TORCH_ACTIVATION_FUNCTIONS.keys(), default="selu")
    weight_init: str = luigi.ChoiceParameter(choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal")
    loss_function: str = "vae_loss"
    dropout_module: str = luigi.ChoiceParameter(choices=TORCH_DROPOUT_MODULES.keys(), default="alpha")

    def create_module(self) -> nn.Module:
        dim = self.n_items \
            if self.project_config.recommender_type == RecommenderType.USER_BASED_COLLABORATIVE_FILTERING \
            else self.n_users
        return VariationalAutoEncoder(dim, self.encoder_layers, self.decoder_layers, dropout_prob=self.dropout_prob,
                                      activation_function=TORCH_ACTIVATION_FUNCTIONS[self.activation_function],
                                      weight_init=TORCH_WEIGHT_INIT[self.weight_init],
                                      dropout_module=TORCH_DROPOUT_MODULES[self.dropout_module])
