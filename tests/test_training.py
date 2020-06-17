#sys.path.insert(0, os.path.dirname(__file__))

import unittest
import luigi
# from luigi import scheduler
# from luigi import server
# import luigi.cmdline
import torch.nn as nn
from mars_gym.model.base_model import LogisticRegression
from mars_gym.simulation.interaction import InteractionTraining
from mars_gym.evaluation.task import EvaluateTestSetPredictions
from unittest.mock import patch
import shutil


class UnitTestInteractionTraining(InteractionTraining):
    def create_module(self) -> nn.Module:
        return LogisticRegression(
            n_factors=10,
            n_users=self.n_users,
            n_items=self.n_items
        )  

@patch("mars_gym.utils.files.OUTPUT_PATH", 'tests/output')
class TestInteractionTraining(unittest.TestCase):
    def setUp(self): 
        shutil.rmtree('tests/output', ignore_errors=True)
    
    def test_training_and_evaluation(self):
        # Training
        job = UnitTestInteractionTraining(project='unittest_interaction_training', epochs=1, test_size=0.1, 
        obs_batch_size=100)
        luigi.build([job], local_scheduler=True)

        # Evaluation
        job = EvaluateTestSetPredictions(model_task_id=job.task_id, model_module="tests.test_training", 
        model_cls='UnitTestInteractionTraining', fairness_columns=[], direct_estimator_module='tests.test_training', 
        direct_estimator_cls='UnitTestInteractionTraining')

        luigi.build([job], local_scheduler=True)


if __name__ == '__main__':
    unittest.main()