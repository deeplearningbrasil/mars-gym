import sys, os

os.environ["OUTPUT_PATH"] = "tests/output"

import unittest
from unittest.mock import patch

import pandas as pd
from unittest.mock import Mock
from mars_gym.data import utils
import luigi

from samples.trivago_simple.data import (
    PrepareInteractionData,
    PrepareMetaData,
    PrepareTrivagoDataFrame,
)
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


# @patch("mars_gym.utils.files.OUTPUT_PATH", "tests/output")
class TestTrivagoRio(unittest.TestCase):
    def setUp(self):
        shutil.rmtree("tests/output", ignore_errors=True)

    # Data Engineer
    def test_data_frame(self):
        job = PrepareTrivagoDataFrame()
        luigi.build([job], local_scheduler=True)

    # Data Simulation
    def test_training(self):
        job_train = SupervisedModelTraining(
            project="samples.trivago_simple.config.trivago_rio",
            recommender_module_class="samples.trivago_simple.simulation.SimpleLinearModel",
            recommender_extra_params={
                "n_factors": 10,
                "metadata_size": 148,
                "window_hist_size": 5,
            },
            epochs=1,
            negative_proportion=0.2,
            test_size=0.1,
        )
        luigi.build([job_train], local_scheduler=True)

    # Data Evaluation
    def test_interactive_and_evaluation(self):
        ## PYTHONPATH="." luigi --module mars_gym.simulation.interaction InteractionTraining --project samples.trivago_simple.config.trivago_rio --recommender-module-class samples.trivago_simple.simulation.SimpleLinearModel --recommender-extra-params '{"n_factors": 10, "metadata_size": 148, "window_hist_size": 5}' --bandit-policy-class samples.trivago_simple.simulation.EGreedyPolicy --bandit-policy-params '{"epsilon": 0.1}' --obs-batch-size 1000
        job_train = InteractionTraining(
            project="samples.trivago_simple.config.trivago_rio",
            recommender_module_class="samples.trivago_simple.simulation.SimpleLinearModel",
            recommender_extra_params={
                "n_factors": 10,
                "metadata_size": 148,
                "window_hist_size": 5,
            },
            bandit_policy_class="samples.trivago_simple.simulation.EGreedyPolicy",
            bandit_policy_params={
                "epsilon": 0.1,
                "seed": 42
            },
            test_size=0.1,
            obs_batch_size=1000,
            num_episodes=1,
        )
        luigi.build([job_train], local_scheduler=True)

    #     luigi.build([job_train], local_scheduler=True)

    #     ## PYTHONPATH="." luigi --module mars_gym.evaluation.task EvaluateTestSetPredictions --model-module samples.exp_trivago_rio.simulation  --model-cls TrivagoModelInteraction --model-task-id TrivagoModelInteraction____epsilon_greedy___epsilon___0_1__d4bfd68660 --fairness-columns "[\"hotel\"]" --no-offpolicy
    #     job_eval = EvaluateTestSetPredictions(
    #         model_task_id=job_train.task_id,
    #         model_task_class="mars_gym.simulation.interaction.InteractionTraining",
    #         fairness_columns=["pos_item_id"],
    #     )
    #     luigi.build([job_eval], local_scheduler=True)

    #     with open(job_eval.output().path + "/metrics.json") as f:
    #         metrics = json.loads(f.read())
    #     print(metrics)
    #     # self.assertEqual(metrics['model_task'], job_train.task_id)
    #     # self.assertEqual(metrics['count'], 2517)
    #     # self.assertEqual(np.round(metrics['precision_at_1'], 2) , 0.12)


if __name__ == "__main__":
    unittest.main()
