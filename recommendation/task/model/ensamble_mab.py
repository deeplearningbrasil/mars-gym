from typing import List
import functools
from itertools import starmap

import json

import luigi
import torch
import torch.nn as nn
from torchbearer import Trial
import torchbearer
import numpy as np
from recommendation.task.model.base import TORCH_ACTIVATION_FUNCTIONS, TORCH_DROPOUT_MODULES
from recommendation.task.model.base import TORCH_WEIGHT_INIT
from recommendation.task.model.interaction import InteractionTraining
from recommendation.task.model.base import BaseTorchModelTraining
from recommendation.rank_metrics import *
from recommendation.task.model.interaction import BanditAgent
from recommendation.data import preprocess_interactions_data_frame
from recommendation.model.bandit import BanditPolicy, BANDIT_POLICIES
from typing import NamedTuple, List, Union, Dict

class IndexMapping(NamedTuple):
    user: Dict[str, int]
    item: Dict[str, int]

class InverseIndexMapping(NamedTuple):
    user: Dict[int, str]
    item: Dict[int, str]

class EnsambleMABInteraction(InteractionTraining):
  user_column: str = luigi.Parameter(default="account_id")
  user_idx_column: str = luigi.Parameter(default="account_idx")
  item_column: str = luigi.Parameter(default="merchant_id")
  item_idx_column: str = luigi.Parameter(default="merchant_idx")

  def create_module(self) -> nn.Module:
    return None
  
  def create_agent(self) -> BanditAgent:
    bandit = BANDIT_POLICIES[self.bandit_policy](reward_model=self.create_module(), 
                                                  index_data=self.index_mapping,
                                                  **self.bandit_policy_params)
    return BanditAgent(bandit)

  def _save_test_set_predictions(self) -> None:
    pass
  
  @property
  def env_data_frame(self) -> pd.DataFrame:
      if not hasattr(self, "_env_data_frame"):
          env_columns = self.obs_columns + [self.project_config.item_column.name]
          if self.project_config.available_arms_column_name:
              env_columns += [self.project_config.available_arms_column_name]

          self._env_data_frame = self.interactions_data_frame.loc[
              self.interactions_data_frame[self.project_config.output_column.name] == 1, env_columns]#.sample(10)
      return self._env_data_frame
      
  @property
  def interactions_data_frame(self) -> pd.DataFrame:
    if not hasattr(self, "_interactions_data_frame"):
        self._interactions_data_frame = preprocess_interactions_data_frame(pd.read_csv(self.test_data_frame_path), self.project_config)
        self._interactions_data_frame.sort_values(self.project_config.timestamp_column_name).reset_index(drop=True)

    return self._interactions_data_frame
      
  @property
  def index_mapping(self) -> IndexMapping:
    
    if not hasattr(self, "_index_mapping"):
      df = pd.concat([pd.read_csv(self.train_data_frame_path), pd.read_csv(self.val_data_frame_path), pd.read_csv(self.test_data_frame_path)])

      user_df = df[[self.user_column, self.user_idx_column]].drop_duplicates()
      user_mapping = pd.Series(index=user_df[self.user_idx_column].values, data=user_df[self.user_column].values)\
          .to_dict()

      item_df = df[[self.item_column, self.item_idx_column]].drop_duplicates()
      item_mapping = pd.Series(index=item_df[self.item_idx_column].values, data=item_df[self.item_column].values) \
          .to_dict()

      self._index_mapping =  IndexMapping(user=user_mapping, item=item_mapping)
    return self._index_mapping 

  # @property
  # def inverse_index_mapping(self) -> InverseIndexMapping:
  #   if not hasattr(self, "_inverse_index_mapping"):
  #       self._inverse_index_mapping = InverseIndexMapping(
  #           user=dict((v, k) for k, v in self.index_mapping.user.items()),
  #           item=dict((v, k) for k, v in self.index_mapping.item.items()),
  #       )
  #   return self._inverse_index_mapping