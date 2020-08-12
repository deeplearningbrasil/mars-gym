import sys, os

os.environ["OUTPUT_PATH"] = "tests/output"

import unittest
from unittest.mock import patch
import pandas as pd
from unittest.mock import Mock
from mars_gym.data import utils
import luigi

from samples.yoochoose.data import (
    LoadAndPrepareDataset,
    InteractionDataFrame,
)  # , PrepareHistoryInteractionData, PrepareInteractionDataFrame
from mars_gym.evaluation.task import EvaluateTestSetPredictions
from mars_gym.simulation.interaction import InteractionTraining
from mars_gym.simulation.training import SupervisedModelTraining
import numpy as np
import json
import os
import mars_gym
from mars_gym.utils import files
from unittest.mock import patch
import shutil
from unittest import mock


class TestYoochoose(unittest.TestCase):
    def setUp(self):
        shutil.rmtree("tests/output", ignore_errors=True)

    # Data Engineer
    def test_prepare_dataset(self):
        job = LoadAndPrepareDataset()
        luigi.build([job], local_scheduler=True)

    def test_data_frame(self):
        job = InteractionDataFrame()
        luigi.build([job], local_scheduler=True)

    # Data Simulation
    def test_training_and_evaluation(self):
        ## PYTHONPATH="." luigi --module mars_gym.simulation.training SupervisedModelTraining --project samples.yoochoose.config.sample_yoochoose_with_negative_sample
        job_train = SupervisedModelTraining(
            project="samples.yoochoose.config.sample_yoochoose_with_negative_sample",
            recommender_module_class="samples.yoochoose.simulation.SimpleLinearModel",
            recommender_extra_params={"n_factors": 10},
            epochs=1,
            negative_proportion=0.2,
            test_size=0.1,
        )
        luigi.build([job_train], local_scheduler=True)

        ## PYTHONPATH="." luigi --module mars_gym.evaluation.task EvaluateTestSetPredictions --model-module samples.yoochoose.simulation  --model-cls YoochoseModelTraining --model-task-id YoochoseModelTraining____model____70b1aa0735 --fairness-columns "[]" --no-offpolicy
        job_eval = EvaluateTestSetPredictions(
            model_task_id=job_train.task_id,
            model_task_class="mars_gym.simulation.training.SupervisedModelTraining",
        )
        luigi.build([job_eval], local_scheduler=True)

        with open(job_eval.output().path + "/metrics.json") as f:
            metrics = json.loads(f.read())

        print(metrics)
        self.assertEqual(metrics["model_task"], job_train.task_id)
        self.assertEqual(metrics["count"], 30643)
        self.assertEqual(np.round(metrics["precision_at_1"], 2), 0.05)

    # Data Evaluation
    def test_interactive_and_evaluation(self):
        ## PYTHONPATH="." luigi --module samples.yoochoose.simulation YoochoseModelInteraction --project test_yoochoose --obs-batch-size 500 --num-episodes 1 --full-refit --bandit-policy random --epochs 100 --sample-size 10000 --seed 47
        job_train = InteractionTraining(
            project="samples.yoochoose.config.sample_yoochoose",
            recommender_module_class="samples.yoochoose.simulation.SimpleLinearModel",
            recommender_extra_params={"n_factors": 10},
            bandit_policy_class="samples.yoochoose.simulation.RandomPolicy",
            bandit_policy_params={"seed": 42},
            batch_size=1,
            epochs=1,
            obs_batch_size=10,
            sample_size=100,
            test_size=0.1,
        )
        luigi.build([job_train], local_scheduler=True)

        ## PYTHONPATH="." luigi --module mars_gym.evaluation.task EvaluateTestSetPredictions --model-module samples.yoochoose.simulation  --model-cls YoochoseModelTraining --model-task-id YoochoseModelTraining____model____70b1aa0735 --fairness-columns "[]" --no-offpolicy
        job_eval = EvaluateTestSetPredictions(
            model_task_id=job_train.task_id,
            model_task_class="mars_gym.simulation.interaction.InteractionTraining",
        )
        luigi.build([job_eval], local_scheduler=True)

        with open(job_eval.output().path + "/metrics.json") as f:
            metrics = json.loads(f.read())
        print(metrics)

        self.assertEqual(metrics["model_task"], job_train.task_id)
        self.assertEqual(metrics["count"], 30643)
        self.assertEqual(np.round(metrics["precision_at_1"], 2), 0.05)


if __name__ == "__main__":
    unittest.main()
