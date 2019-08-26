import abc
import importlib

import luigi

from recommendation.task.model.base import BaseTorchModelTraining, load_torch_model_training_from_task_id
from recommendation.task.data_preparation.base import WINDOW_FILTER_DF


class BaseEvaluationTask(luigi.Task, metaclass=abc.ABCMeta):
    model_module: str = luigi.Parameter(default="datalife.task.model.matrix_factorization")
    model_cls: str = luigi.Parameter(default="MatrixFactorizationTraining")
    model_task_id: str = luigi.Parameter()
    window_filter: str = luigi.ChoiceParameter(choices=WINDOW_FILTER_DF.keys(), default="one_week")

    @property
    def model_training(self) -> BaseTorchModelTraining:
        if not hasattr(self, "_model_training"):
            module = importlib.import_module(self.model_module)
            class_ = getattr(module, self.model_cls)

            self._model_training = load_torch_model_training_from_task_id(class_, self.model_task_id)

        return self._model_training