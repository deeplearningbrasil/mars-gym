import sys, os

os.environ["OUTPUT_PATH"] = "tests/output"

import unittest
import luigi
import torch.nn as nn
from mars_gym.model.base_model import LogisticRegression
from mars_gym.simulation.interaction import InteractionTraining
from mars_gym.simulation.training import SupervisedModelTraining
from mars_gym.evaluation.task import EvaluateTestSetPredictions
from unittest.mock import patch
import shutil


class TestTraining(unittest.TestCase):
    def setUp(self):
        shutil.rmtree("tests/output", ignore_errors=True)

    def test_Interaction_training_and_evaluation(self):
        # Training
        job = InteractionTraining(
            project="tests.factories.config.test_base_training",
            recommender_module_class="mars_gym.model.base_model.LogisticRegression",
            recommender_extra_params={"n_factors": 10},
            epochs=1,
            test_size=0.1,
            obs_batch_size=100,
        )
        luigi.build([job], local_scheduler=True)

        # Evaluation
        #PYTHONPATH="." luigi --module mars_gym.evaluation.task EvaluateTestSetPredictions --model-task-id InteractionTraining____samples_trivago____epsilon___0_1__4fc1370d9d --model-task-class "mars_gym.simulation.interaction.InteractionTraining"
        job = EvaluateTestSetPredictions(
            model_task_id=job.task_id,
            model_task_class="mars_gym.simulation.interaction.InteractionTraining",
        )

        luigi.build([job], local_scheduler=True)

    def test_batch_training_and_evaluation(self):
        # Training
        job = SupervisedModelTraining(
            project="tests.factories.config.test_base_training",
            recommender_module_class="mars_gym.model.base_model.LogisticRegression",
            recommender_extra_params={"n_factors": 10},
            epochs=10,
            test_size=0.1,
        )
        luigi.build([job], local_scheduler=True)

        # Evaluation

        job = EvaluateTestSetPredictions(
            model_task_id=job.task_id,
            model_task_class="mars_gym.simulation.training.SupervisedModelTraining",
        )

        luigi.build([job], local_scheduler=True)


if __name__ == "__main__":
    unittest.main()
