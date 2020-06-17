import sys, os
import unittest
from unittest.mock import patch

import pandas as pd
from unittest.mock import Mock
from mars_gym.data import utils
import luigi

from samples.exp_trivago_rio.data import PrepareMetaData, PrepareHistoryInteractionData, PrepareTrivagoDataFrame
from samples.exp_trivago_rio.simulation import TrivagoModelTraining, TrivagoModelInteraction
from mars_gym.evaluation.task import EvaluateTestSetPredictions
import numpy as np
import json
import os
import mars_gym
from mars_gym.utils import files
from unittest.mock import patch
import shutil



#@patch("samples.exp_trivago_rio.data.BASE_DIR", 'tests/output/trivago_rio')
#@patch("samples.exp_trivago_rio.data.DATASET_DIR", 'tests/output/trivago_rio/dataset')
#@patch("samples.exp_trivago_rio.data.OUTPUT_PATH", 'tests/output')
@patch("mars_gym.utils.files.OUTPUT_PATH", 'tests/output')
class TestTrivagoRio(unittest.TestCase):
  def setUp(self): 
    shutil.rmtree('tests/output', ignore_errors=True)

  # Data Engineer
  def test_prepare_meta_data(self):
    job = PrepareMetaData()
    luigi.build([job], local_scheduler=True)

  def test_prepare_history_interaction_data(self):
    job = PrepareHistoryInteractionData()
    luigi.build([job], local_scheduler=True)

  def test_data_frame(self):
    job = PrepareTrivagoDataFrame()
    luigi.build([job], local_scheduler=True)

  # Data Simulation
  def test_training(self):
    ## PYTHONPATH="." luigi --module samples.exp_trivago_rio.simulation TrivagoModelTraining --project trivago_rio
    job = TrivagoModelTraining(project='trivago_rio')
    luigi.build([job], local_scheduler=True)


  # Data Evaluation
  def test_interactive_and_evaluation(self):
    ## PYTHONPATH="." luigi --module samples.exp_trivago_rio.simulation TrivagoModelInteraction --project trivago_rio --n-factors 100 --metrics '["loss"]'  --obs-batch-size 1000 --batch-size 200 --num-episodes 1 --val-split-type random --full-refit --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.1}' --epochs 100 --seed 42
    job_train = TrivagoModelInteraction(project='trivago_rio', n_factors=100, batch_size=200, epochs=1,  num_episodes=1, obs_batch_size=1000, bandit_policy='epsilon_greedy')
    luigi.build([job_train], local_scheduler=True)
    
    ## PYTHONPATH="." luigi --module mars_gym.evaluation.task EvaluateTestSetPredictions --model-module samples.exp_trivago_rio.simulation  --model-cls TrivagoModelInteraction --model-task-id TrivagoModelInteraction____epsilon_greedy___epsilon___0_1__d4bfd68660 --fairness-columns "[\"hotel\"]" --no-offpolicy
    job_eval = EvaluateTestSetPredictions(model_module='samples.exp_trivago_rio.simulation', model_cls='TrivagoModelInteraction', 
          model_task_id=job_train.task_id, fairness_columns=["pos_item_id"])
    luigi.build([job_eval], local_scheduler=True)

    with open(job_eval.output().path+"/metrics.json") as f: 
      metrics = json.loads(f.read()) 
    print(metrics)
    #self.assertEqual(metrics['model_task'], job_train.task_id)
    #self.assertEqual(metrics['count'], 2517)    
    #self.assertEqual(np.round(metrics['precision_at_1'], 2) , 0.12)
    

if __name__ == '__main__':
    unittest.main()