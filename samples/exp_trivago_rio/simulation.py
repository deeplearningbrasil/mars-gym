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
        metadata_size: int,
        window_hist_size: int,
    ):
        super(SimpleLinearModel, self).__init__()

        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)

        num_dense = (
            n_factors
            + window_hist_size
            + 1
            + metadata_size
        )  # + n_factors * n_factors

        self.dense = nn.Sequential(
            nn.Linear(num_dense, int(num_dense / 2)),
            nn.SELU(),
            nn.Linear(int(num_dense / 2), 1),
        )

    def flatten(self, input):
        return input.view(input.size(0), -1)

    def normalize(self, x):
        x = F.normalize(x, p=2, dim=1)
        return x

    def item_dot_history(self, itemA, itemB):
        dot = torch.matmul(
            self.normalize(itemA.unsqueeze(1)), self.normalize(itemB.permute(0, 2, 1))
        )
        return self.flatten(dot)

    def forward(
        self,
        user_ids,
        item_ids,
        pos_item_id,
        list_reference_item,
        list_metadata,
      ):
        # Geral embs
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)

        # Item embs
        interaction_item_emb = self.item_embeddings(list_reference_item)

        # Dot Item X History
        item_dot_interaction_item_emb = self.item_dot_history(item_emb, interaction_item_emb)
        #raise(Exception(user_emb.shape, interaction_item_emb.shape, item_dot_interaction_item_emb.shape))
        x = torch.cat(
            (
                item_emb,
                item_dot_interaction_item_emb,
                pos_item_id.float().unsqueeze(1),
                list_metadata.float(),
            ),
            dim=1,
        )

        x   = self.dense(x)
        out = torch.sigmoid(x)
        return out

class TrivagoModelTrainingMixin(object):
    n_factors: int = luigi.IntParameter(default=128)

    def create_module(self) -> nn.Module:
        return SimpleLinearModel(
            n_users=max(self.index_mapping[self.project_config.user_column.name].values()) + 1,
            n_items=max(self.index_mapping[self.project_config.item_column.name].values()) + 1,
            n_factors=self.n_factors,
            metadata_size=148,
            window_hist_size=5,
        )

class TrivagoModelInteraction(TrivagoModelTrainingMixin, InteractionTraining):
    pass

class TrivagoModelTraining(TrivagoModelTrainingMixin, BaseTorchModelWithAgentTraining):
    pass
