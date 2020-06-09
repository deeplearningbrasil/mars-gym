#import sys, os
import pandas as pd
import luigi
import torch
import torch.nn as nn
from mars_gym.model.base_model import LogisticRegression
from mars_gym.data.dataset import RandomData
from mars_gym.task.model.interaction import InteractionTraining
from mars_gym.task.data.base import (
    BasePySparkTask,
    BasePrepareDataFrames,
)


class TestDataFrames(BasePrepareDataFrames):
    @property
    def timestamp_property(self) -> str:
        return "timestamp"

    def dataset_dir(self) -> str:
        return os.path.join("tests", "output", "test")

    def read_data_frame(self) -> pd.DataFrame:
        return RandomData().data()

class TestInteractionTraining(InteractionTraining):
    def create_module(self) -> nn.Module:
        return LogisticRegression(
          n_factors=10
        )  

class UnitTestInteractionTraining(unittest.TestCase):
  def setUp(self): 
    self._random_data = RandomData()
    pass
      
  def test_training(self):
    #job = TestInteractionTraining(project='test_interaction_training')
    
    df = self._random_data.data()
    print(df)
    
    #luigi.build([job], local_scheduler=True)
    
    #self.assertEqual(len(config.arms), 2)
