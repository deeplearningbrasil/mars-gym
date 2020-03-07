import functools
import json
import os
import pprint
from itertools import starmap
from multiprocessing.pool import Pool
from time import time
from typing import Dict, Tuple, List, Any, Union

import luigi
import math
import numpy as np
import pandas as pd
import torch
from sklearn import manifold
from torch.utils.data.dataset import Dataset
from torchbearer import Trial
from tqdm import tqdm

from recommendation.task.model.base import BaseEvaluationTask
from recommendation.files import  get_simulator_datalog_path, get_interator_datalog_path, get_ground_truth_datalog_path

class InteractionEvaluation(BaseEvaluationTask):

  def output(self):
    return luigi.LocalTarget(os.path.join("output", "evaluation", self.__class__.__name__, "results", self.task_name))

  def geral_stats(self, df):
    pass


  def run(self):
    module   = self.model_training#.get_trained_module()
    
    inter_df = pd.read_csv(get_interator_datalog_path(module.output().path))
    sim_df   = pd.read_csv(get_simulator_datalog_path(module.output().path))
    gt_df    = pd.read_csv(get_ground_truth_datalog_path(module.output().path))

    print(inter_df.head())
    print(sim_df.head())
    print(gt_df.head())