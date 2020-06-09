import sys, os
#sys.path.insert(0, os.path.dirname(__file__))

import unittest
import pandas as pd
from unittest.mock import Mock
import luigi
# from luigi import scheduler
# from luigi import server
# import luigi.cmdline
import torch
import torch.nn as nn
from mars_gym.model.base_model import LogisticRegression
from mars_gym.data.dataset import RandomData
from mars_gym.task.model.interaction import InteractionTraining
from mars_gym.task.data.base import (
    BasePySparkTask,
    BasePrepareDataFrames,
)


class UnitTestInteractionTraining(InteractionTraining):
    def create_module(self) -> nn.Module:
        return LogisticRegression(
            n_factors=10,
            n_users=self.n_users,
            n_items=self.n_items
        )  

class TestInteractionTraining(unittest.TestCase):
  def setUp(self): 
    self._random_data = RandomData()
    pass
      
  def test_training(self):
    job = UnitTestInteractionTraining(project='unittest_interaction_training')
    luigi.build([job], local_scheduler=True)
    
    #self.assertEqual(len(config.arms), 2)

if __name__ == '__main__':
    unittest.main()


# PYTHONPATH="." luigi --module mars_gym.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}' --epochs 1

# PYTHONPATH="." luigi --module mars_gym.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}' --epochs 1 --test-size 0.1

# PYTHONPATH="." luigi --module tests.test_training UnitTestInteractionTraining --project unittest_interaction_training --epochs 1 --test-size 0.1

# PYTHONPATH="." luigi --module mars_gym.task.evaluation.evaluation EvaluateTestSetPredictions --model-module tests.test_training  \
#  --model-cls UnitTestInteractionTraining --model-task-id UnitTestInteractionTraining____model____68f30a0e4b --fairness-columns "[]" --direct-estimator-module tests.test_training --direct-estimator-cls UnitTestInteractionTraining     