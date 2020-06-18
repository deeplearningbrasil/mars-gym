# sys.path.insert(0, os.path.dirname(__file__))

import unittest
import luigi

# from luigi import scheduler
# from luigi import server
# import luigi.cmdline
import torch.nn as nn
from mars_gym.model.base_model import LogisticRegression
from mars_gym.simulation.interaction import InteractionTraining
from mars_gym.simulation.training import TorchModelWithAgentTraining
from mars_gym.evaluation.task import EvaluateTestSetPredictions
from unittest.mock import patch
import shutil


class TestTraining(unittest.TestCase):
    def setUp(self):
        shutil.rmtree("tests/output", ignore_errors=True)

    @patch("mars_gym.utils.files.OUTPUT_PATH", "tests/output")
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
        job = EvaluateTestSetPredictions(
            model_task_id=job.task_id,
            model_task_class='mars_gym.simulation.interaction.InteractionTraining')

        luigi.build([job], local_scheduler=True)

    @patch("mars_gym.utils.files.OUTPUT_PATH", "tests/output")
    def test_batch_training_and_evaluation(self):
        # Training
        job = TorchModelWithAgentTraining(
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
            model_task_class='mars_gym.simulation.interaction.TorchModelWithAgentTraining')

        luigi.build([job], local_scheduler=True)

if __name__ == '__main__':
    unittest.main()
