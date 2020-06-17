from multiprocessing import Pool
from typing import Union, Tuple, Optional, List
import os
import json

import luigi
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchbearer import Trial

from mars_gym.simulation.base import (
    TORCH_ACTIVATION_FUNCTIONS,
    TORCH_DROPOUT_MODULES,
    TORCH_LOSS_FUNCTIONS,
    TORCH_WEIGHT_INIT,
    BaseTorchModelWithAgentTraining
)
from mars_gym.simulation.interaction import InteractionTraining


class SimpleLinearModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
    ):
        super(SimpleLinearModel, self).__init__()

        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)
        self.dayofweek_embeddings       = nn.Embedding(7, n_factors)

        num_dense  = n_factors * (2 + 5) + 1 
        
        self.dense = nn.Sequential(
            nn.Linear(num_dense, int(num_dense / 2)),
            nn.SELU(),
            nn.Linear(int(num_dense / 2), 1),
        )

    def flatten(self, input):
        return input.view(input.size(0), -1)

    def forward(
        self,
        session_ids,
        item_ids,
        dayofweek,
        step,
        item_history_ids
      ):
        # Item emb
        item_emb = self.item_embeddings(item_ids)

        # Item History embs
        interaction_item_emb = self.flatten(self.item_embeddings(item_history_ids))

        # DayofWeek Emb
        dayofweek_emb = self.dayofweek_embeddings(dayofweek.long())

        x = torch.cat(
            (
                item_emb,
                interaction_item_emb,
                dayofweek_emb,
                step.float().unsqueeze(1),
            ),
            dim=1,
        )

        out = torch.sigmoid(self.dense(x))
        return out


class YoochoseModelTrainingMixin(object):
    n_factors: int = luigi.IntParameter(default=128)
    learning_rate: float = luigi.FloatParameter(1e-4)
    test_size: float = luigi.FloatParameter(default=0.1)
    loss_function: str = luigi.ChoiceParameter(
        choices=TORCH_LOSS_FUNCTIONS.keys(), default="bce"
    )

    def create_module(self) -> nn.Module:
        return SimpleLinearModel(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors
        )


class YoochoseModelInteraction(YoochoseModelTrainingMixin, InteractionTraining):
    pass

class YoochoseModelTraining(YoochoseModelTrainingMixin, BaseTorchModelWithAgentTraining):
    negative_proportion: int = luigi.FloatParameter(0.8)
    pass
