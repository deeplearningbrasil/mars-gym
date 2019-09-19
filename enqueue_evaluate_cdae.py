from typing import List

import luigi


# - Para múltiplas GPUs:
## ./cuda_luigid --background --pidfile /tmp/luigid.pid --logdir /tmp/luigid_log
## PYTHONPATH="." ./cuda_luigi --module enqueue_pneumonia EnqueuePneumonia --seed 42 --workers 4
# - Para uma única GPU:
## PYTHONPATH="." luigi --module enqueue_pneumonia EnqueuePneumonia --seed 42 --local-scheduler
from recommendation.task.ifood import EvaluateIfoodCDAEModel


class EnqueueEvaluateCDAE(luigi.WrapperTask):
    task_ids: List[str] = luigi.ListParameter()

    def requires(self):
        for task_id in self.task_ids:
            yield EvaluateIfoodCDAEModel(
                model_module="recommendation.task.model.auto_encoder",
                model_cls="UnconstrainedAutoEncoderTraining",
                model_task_id=task_id,
            )

