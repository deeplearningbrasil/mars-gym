import abc
import importlib

import luigi

from recommendation.task.model.base import BaseTorchModelTraining, load_torch_model_training_from_task_id


class BaseEvaluationTask(luigi.Task, metaclass=abc.ABCMeta):
    model_module: str = luigi.Parameter(default="datalife.task.model.matrix_factorization")
    model_cls: str = luigi.Parameter(default="MatrixFactorizationTraining")
    model_task_id: str = luigi.Parameter()

    @property
    def task_name(self):
        return self.model_task_id + "_" + self.task_id.split("_")[-1]

    @property
    def model_training(self) -> BaseTorchModelTraining:
        if not hasattr(self, "_model_training"):
            module = importlib.import_module(self.model_module)
            class_ = getattr(module, self.model_cls)

            self._model_training = load_torch_model_training_from_task_id(class_, self.model_task_id)

        return self._model_training