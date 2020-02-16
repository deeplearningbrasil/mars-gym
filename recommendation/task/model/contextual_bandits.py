import luigi
import torch.nn as nn
import os
import numpy as np
from typing import Callable, List, Type, Union

from recommendation.task.model.base import BaseTorchModelTraining, TORCH_LOSS_FUNCTIONS, TORCH_ACTIVATION_FUNCTIONS, TORCH_WEIGHT_INIT, \
    TORCH_DROPOUT_MODULES
from recommendation.model.contextual_bandits import ContextualBandit, DirectEstimator
from recommendation.task.model.base import TORCH_WEIGHT_INIT

class DirectEstimatorTraining(BaseTorchModelTraining):
    n_factors: int = luigi.IntParameter(default=100)
    dropout_prob: float = luigi.FloatParameter(default=0.1)
    learning_rate: float = luigi.FloatParameter(1e-4)
    loss_function: str = luigi.ChoiceParameter(choices=TORCH_LOSS_FUNCTIONS.keys(), default="bce")
    epochs: int = luigi.IntParameter(default=50)

    def create_module(self) -> nn.Module:
        return DirectEstimator(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            dropout_prob=self.dropout_prob)

class ContextualBanditsTraining(BaseTorchModelTraining):
    loss_function: str = luigi.ChoiceParameter(choices=["crm"], default="crm")
    n_factors: int = luigi.IntParameter(default=128)
    weight_init: str = luigi.ChoiceParameter(choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal")
    dropout_module: str = luigi.ChoiceParameter(choices=TORCH_DROPOUT_MODULES.keys(), default="alpha")
    activation_function: str = luigi.ChoiceParameter(choices=TORCH_ACTIVATION_FUNCTIONS.keys(), default="selu")
    content_layers: List[int] = luigi.ListParameter(default=[200, 128])
    use_original_content: bool = luigi.BoolParameter(default=False)
    use_buys_visits: bool = luigi.BoolParameter(default=False)
    user_embeddings: bool = luigi.BoolParameter(default=False)
    item_embeddings: bool = luigi.BoolParameter(default=False)
    context_embeddings: bool = luigi.BoolParameter(default=False)
    use_numerical_content: bool = luigi.BoolParameter(default=False)
    use_textual_content: bool = luigi.BoolParameter(default=False)
    use_normalize: bool = luigi.BoolParameter(default=False)
    binary: bool = luigi.BoolParameter(default=False)
    predictor: str = luigi.ChoiceParameter(choices=["simple_logistic_regression", "logistic_regression", "factorization_machine"], default="logistic_regression")
    weight_init: str = luigi.ChoiceParameter(choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal")
    activation_function: str = luigi.ChoiceParameter(choices=TORCH_ACTIVATION_FUNCTIONS.keys(), default="selu")
    word_embeddings_size: int = luigi.IntParameter(default=128)
    fm_order: int = luigi.IntParameter(default=1)
    fm_hidden_layers: List[int] = luigi.ListParameter(default=[64,32])
    fm_deep: bool = luigi.BoolParameter(default=False)


    @property
    def non_textual_input_dim(self):
        if not hasattr(self, "_non_textual_input_dim"):
            self._non_textual_input_dim = int(self.metadata_data_frame.iloc[0]["non_textual_input_dim"])
        return self._non_textual_input_dim

    @property
    def vocab_size(self):
        if not hasattr(self, "_vocab_size"):
            self._vocab_size = int(self.metadata_data_frame.iloc[0]["vocab_size"])
        return self._vocab_size

    @property
    def menu_full_text_max_words(self):
        if not hasattr(self, "_menu_full_text_max_words"):
            self._menu_full_text_max_words = int(self.metadata_data_frame.iloc[0]["menu_full_text_max_words"])
        return self._menu_full_text_max_words

    @property
    def description_max_words(self):
        if not hasattr(self, "_description_max_words"):
            self._description_max_words = int(self.metadata_data_frame.iloc[0]["description_max_words"])
        return self._description_max_words

    @property
    def category_names_max_words(self):
        if not hasattr(self, "_category_names_max_words"):
            self._category_names_max_words = int(self.metadata_data_frame.iloc[0]["category_names_max_words"])
        return self._category_names_max_words

    @property
    def name_max_words(self):
        if not hasattr(self, "_name_max_words"):
            self._name_max_words = int(self.metadata_data_frame.iloc[0]["trading_name_max_words"])
        return self._name_max_words

    def create_module(self) -> nn.Module:
        return ContextualBandit(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            use_original_content=self.use_original_content,
            use_buys_visits=self.use_buys_visits,
            user_embeddings=self.user_embeddings,
            item_embeddings=self.item_embeddings,
            context_embeddings=self.context_embeddings,
            use_numerical_content=self.use_numerical_content,
            numerical_content_dim=self.non_textual_input_dim,
            use_textual_content=self.use_textual_content,
            vocab_size=self.vocab_size,
            word_embeddings_size=self.word_embeddings_size,
            use_normalize=self.use_normalize,
            content_layers=self.content_layers,
            binary=self.binary,
            activation_function=self.activation_function,
            predictor=self.predictor,
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
            fm_deep=self.fm_deep,
            fm_order=self.fm_order,
            fm_hidden_layers=self.fm_hidden_layers,
        )