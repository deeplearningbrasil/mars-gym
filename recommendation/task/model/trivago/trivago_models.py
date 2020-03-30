from typing import List
import functools
from itertools import starmap
import os
import json

import luigi
import torch
import torch.nn as nn
from torchbearer import Trial
import torchbearer
import numpy as np
from recommendation.model.trivago.trivago_models import TestModel, SimpleCNNModel, SimpleRNNModel, SimpleLinearModel, SimpleCNNTransformerModel
from recommendation.task.model.base import TORCH_ACTIVATION_FUNCTIONS, TORCH_DROPOUT_MODULES
from recommendation.task.model.base import TORCH_WEIGHT_INIT
from recommendation.task.model.interaction import InteractionTraining
from recommendation.task.model.base import BaseTorchModelTraining
from recommendation.rank_metrics import *


class TrivagoModelTrainingMixin(object):
  loss_function: str = luigi.ChoiceParameter(choices=["crm", "bce"], default="crm")
  n_factors: int = luigi.IntParameter(default=128)
  weight_init: str = luigi.ChoiceParameter(choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal")
  dropout_prob: float = luigi.FloatParameter(default=0.1)
  dropout_module: str = luigi.ChoiceParameter(choices=TORCH_DROPOUT_MODULES.keys(), default="alpha")
  activation_function: str = luigi.ChoiceParameter(choices=TORCH_ACTIVATION_FUNCTIONS.keys(), default="selu")
  filter_sizes: List[int] = luigi.ListParameter(default=[1, 3, 5])
  num_filters: int = luigi.IntParameter(default=64)

  @property
  def window_hist_size(self):
      if not hasattr(self, "_window_hist_size"):
          self._window_hist_size = int(self.train_data_frame.iloc[0]["window_hist_size"])
      return self._window_hist_size

  @property
  def metadata_size(self):
      if not hasattr(self, "_meta_data_size"):
          self._meta_data_size = int(self.metadata_data_frame.shape[1] - 3)
      return self._meta_data_size     


  def create_module(self) -> nn.Module:

      return SimpleCNNModel(
          window_hist_size=self.window_hist_size,
          vocab_size=self.vocab_size,
          metadata_size=self.metadata_size,
          n_users=self.n_users,
          n_items=self.n_items,
          n_factors=self.n_factors,
          filter_sizes=self.filter_sizes,
          num_filters=self.num_filters,
          dropout_prob=self.dropout_prob,
          dropout_module=TORCH_DROPOUT_MODULES[self.dropout_module],            
      )


class TrivagoModelInteraction(TrivagoModelTrainingMixin, InteractionTraining):
  pass
        
class TrivagoModelTraining(TrivagoModelTrainingMixin, BaseTorchModelTraining):

  def evaluate(self):
      module      = self.get_trained_module()
      val_loader  = self.get_val_generator()

      print("================== Evaluate ========================")
      trial = Trial(module, self._get_optimizer(module), self._get_loss_function(), callbacks=[],
                    metrics=self.metrics).to(self.torch_device)\
                  .with_test_generator(val_loader)#.eval()

      scores_tensor: Union[torch.Tensor, Tuple[torch.Tensor]] = trial.predict(verbose=2)
      scores: np.ndarray = scores_tensor.detach().cpu().numpy().reshape(-1)
      
      df_eval          = self.val_data_frame
      df_eval['score'] = scores

      group = df_eval.sample(frac=1)\
          .groupby(['timestamp', 'user_idx', 'session_idx', 'step'])
      
      df_eval                  = group.agg({'impressions': 'first'})
      df_eval['list_item_idx'] = group['item_idx'].apply(list)
      df_eval['list_score']    = group['score'].apply(list)
      df_eval['pos_item_idx']  = group['pos_item_idx'].apply(list)
      df_eval['clicked']       = group['clicked'].apply(list)
      df_eval['item_idx']      = df_eval.apply(lambda row: int(np.max(np.array(row['clicked'])*np.array(row['list_item_idx']))), axis=1)
      
      def sort_and_bin(row):
          list_sorted, score = zip(*sorted(zip(row['list_item_idx'], row['list_score']), key = lambda x: x[1], reverse=True))
          list_sorted = (np.array(list_sorted) == row['item_idx']).astype(int)
          
          return list(list_sorted)
      
      df_eval['sorted_list'] = df_eval.apply(sort_and_bin, axis=1)
      df_eval = df_eval.reset_index()

      df_eval.head()

      metric = {
          'reciprocal_rank@5':  np.mean([reciprocal_rank_at_k(l, 5) for l in list(df_eval['sorted_list'])] ),
          'precision@1':        np.mean([precision_at_k(l, 1) for l in list(df_eval['sorted_list'])] ),
          'ndcg@5':             np.mean([ndcg_at_k(l, 5) for l in list(df_eval['sorted_list'])] ),
          'MRR':                mean_reciprocal_rank(list(df_eval['sorted_list']))
      }

      with open(os.path.join(self.output().path, "metric.json"), "w") as params_file:
          json.dump(metric, params_file, default=lambda o: dict(o), indent=4)
    
      print(json.dumps(metric, indent = 4))
      df_eval.to_csv(os.path.join(self.output().path, "df_eval.csv"))
