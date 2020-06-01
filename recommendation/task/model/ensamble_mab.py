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
from typing import List, Tuple, Dict, Any, Union, Optional
from recommendation.task.data_preparation.new_ifood import CheckDataset

class IndexMapping(NamedTuple):
    user: Dict[str, int]
    item: Dict[str, int]

class InverseIndexMapping(NamedTuple):
    user: Dict[int, str]
    item: Dict[int, str]

class EnsambleMABInteraction(InteractionTraining):
  dataset_split_method: str = luigi.ChoiceParameter(choices=["holdout", "time", "column", "k_fold"], default="time")
  user_column: str = luigi.Parameter(default="account_id")
  user_idx_column: str = luigi.Parameter(default="account_idx")
  item_column: str = luigi.Parameter(default="merchant_id")
  item_idx_column: str = luigi.Parameter(default="merchant_idx")
  
  def create_agent(self) -> BanditAgent:
    self.seed_everything()
    bandit = BANDIT_POLICIES[self.bandit_policy](reward_model=self.create_module(), 
                                                  seed=self.seed,
                                                  index_data=self.index_mapping,
                                                  **self.bandit_policy_params)
    return BanditAgent(bandit)

  def create_module(self) -> nn.Module:
    return None

  def _save_test_set_predictions(self) -> None:
    pass
  
  @property
  def metadata_data_frame_path(self) -> Optional[str]:
    return None
      
  @property
  def interactions_data_frame(self) -> pd.DataFrame:
    if not hasattr(self, "_interactions_data_frame"):
        self._interactions_data_frame = preprocess_interactions_data_frame(pd.read_csv(self.val_data_frame_path), self.project_config)
        #self._interactions_data_frame['item_metadata'] = 1
        self._interactions_data_frame.sort_values(['order_date_local', 'shift']).reset_index(drop=True)

    return self._interactions_data_frame
            
  @property
  def index_mapping(self) -> IndexMapping:
    if not hasattr(self, "_index_mapping"):
      user_df  = pd.read_csv(CheckDataset().output()[3].path)
      item_df  = pd.read_csv(CheckDataset().output()[4].path)

      user_mapping = pd.Series(index=user_df[self.user_idx_column].values, data=user_df[self.user_column].values)\
          .to_dict()

      item_mapping = pd.Series(index=item_df[self.item_idx_column].values, data=item_df[self.item_column].values) \
          .to_dict()

      self._index_mapping =  IndexMapping(user=user_mapping, item=item_mapping)
    return self._index_mapping 