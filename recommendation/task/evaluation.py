import abc
import importlib
import os
import luigi
import json
from recommendation.files import get_params_path

from recommendation.task.model.base import BaseTorchModelTraining, load_torch_model_training_from_task_id


class BaseEvaluationTask(luigi.Task, metaclass=abc.ABCMeta):
    model_module: str = luigi.Parameter(default="recommendation.task.model.matrix_factorization")
    model_cls: str = luigi.Parameter(default="MatrixFactorizationTraining")
    model_task_id: str = luigi.Parameter()
    limit_list_size: int = luigi.IntParameter(default=50)

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

    @property
    def n_items(self):
        return self.model_training.n_items

    @property
    def output_path(self):
        return os.path.join("output", "evaluation", self.__class__.__name__, "results", self.task_name)

    def _save_params(self):
        with open(get_params_path(self.output_path), "w") as params_file:
            json.dump(self.param_kwargs, params_file, default=lambda o: dict(o), indent=4)