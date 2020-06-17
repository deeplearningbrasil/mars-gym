import sys, os
import unittest
import pandas as pd
from unittest.mock import Mock
from mars_gym.data import utils
import luigi
from unittest.mock import patch
import shutil

from mars_gym.data.utils import DownloadDataset


@patch("mars_gym.utils.files.OUTPUT_PATH", "tests/output")
class TestDataset(unittest.TestCase):
    def setUp(self):
        shutil.rmtree("tests/output", ignore_errors=True)

    def test_load_dataset(self):
        df_trivago = utils.load_dataset("trivago_rio", output_path="tests/output")
        df_random = utils.load_dataset("random", output_path="tests/output")

        self.assertEqual(len(df_trivago), 2)
        self.assertEqual(len(df_random), 1)

    def test_download_dataset_task(self):
        job = DownloadDataset(dataset="trivago_rio")
        luigi.build([job], local_scheduler=True)


if __name__ == "__main__":
    unittest.main()
