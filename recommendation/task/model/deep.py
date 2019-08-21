from typing import List

import torch.nn as nn

import luigi

from recommendation.model.deep import DeepCTR
from recommendation.task.meta_config import RecommenderType
from recommendation.task.model.base import BaseTorchModelTraining, TORCH_ACTIVATION_FUNCTIONS, TORCH_WEIGHT_INIT, \
    TORCH_DROPOUT_MODULES
from recommendation.torch import MaskedZeroesLoss
import pandas as pd

class DeepTraining(BaseTorchModelTraining):
    input_d_dim: int = luigi.IntParameter(default=100)
    input_c_dim: int = luigi.IntParameter(default=100)
    n_factors:   int = luigi.IntParameter(default=1)

    dense_layers: List[int] = luigi.ListParameter(default=[1024, 512, 256])

    dropout_prob: float = luigi.FloatParameter(default=None)
    activation_function: str = luigi.ChoiceParameter(choices=TORCH_ACTIVATION_FUNCTIONS.keys(), default="selu")
    weight_init: str = luigi.ChoiceParameter(choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal")
    dropout_module: str = luigi.ChoiceParameter(choices=TORCH_DROPOUT_MODULES.keys(), default="alpha")


    def create_module(self) -> nn.Module:
        print(self.size_embs)
        return DeepCTR(input_d_dim=self.input_d_dim, input_c_dim=self.input_c_dim, 
                       n_factors=self.n_factors,
                       size_embs=list(self.size_embs.values()),
                       dense_layers=self.dense_layers,
                       dropout_prob=self.dropout_prob,
                       activation_function=TORCH_ACTIVATION_FUNCTIONS[self.activation_function],
                       weight_init=TORCH_WEIGHT_INIT[self.weight_init],
                       dropout_module=TORCH_DROPOUT_MODULES[self.dropout_module])

    @property
    def size_embs(self):
        columns = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']

        if not hasattr(self, "_size_embs"):
            self._size_embs = self.build_n_categories(columns)
        return self._size_embs

            
    def build_n_categories(self, columns):
        train_df = pd.read_csv(self.input()[0].path)
        val_df   = pd.read_csv(self.input()[1].path)

        df       = pd.concat([train_df, val_df])

        unique_vals = {}

        for c in columns:
            unique_vals[c] = len(df[c].unique())

        return unique_vals