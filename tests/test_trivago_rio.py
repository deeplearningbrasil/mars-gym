import sys, os
import unittest
import pandas as pd
from unittest.mock import Mock
from mars_gym.data import utils
import luigi

from exp_trivago_rio.data import PrepareMetaData, PrepareHistoryInteractionData, PrepareTrivagoDataFrame
from exp_trivago_rio.simulation import TrivagoModelTraining, TrivagoModelInteraction

import mars_gym
class TestTrivagoRio(unittest.TestCase):

  def test_prepare_meta_data(self):
    job = PrepareMetaData()
    luigi.build([job], local_scheduler=True)

  def test_prepare_history_interaction_data(self):
    job = PrepareHistoryInteractionData()
    luigi.build([job], local_scheduler=True)

  def test_data_frame(self):
    job = PrepareTrivagoDataFrame()
    luigi.build([job], local_scheduler=True)

  #PYTHONPATH="." luigi --module exp_trivago_rio.simulation TrivagoModelTraining --project trivago_rio
  def test_training(self):
    job = TrivagoModelTraining(project='trivago_rio')
    luigi.build([job], local_scheduler=True)

  def test_interactive(self):
    job = TrivagoModelInteraction(project='trivago_rio', obs_batch_size=1000, bandit_policy='epsilon_greedy')
    luigi.build([job], local_scheduler=True)



if __name__ == '__main__':
    unittest.main()