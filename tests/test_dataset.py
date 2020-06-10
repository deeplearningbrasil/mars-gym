import sys, os
import unittest
import pandas as pd
from unittest.mock import Mock
from mars_gym.data import utils
import luigi

from mars_gym.data.utils import DownloadDataset

class TestDataset(unittest.TestCase):

  def test_get_data_home(self):
    path = utils.get_data_home()
    self.assertEqual(path, "output")

  def test_load_dataset(self):
    df_trivago = utils.load_dataset("trivago_rio")
    df_random  = utils.load_dataset("random")
    
    self.assertEqual(len(df_trivago), 2)    
    self.assertEqual(len(df_random), 1)

  def test_download_dataset_task(self):
    job = DownloadDataset(dataset='trivago_rio')
    luigi.build([job], local_scheduler=True)


if __name__ == '__main__':
    unittest.main()