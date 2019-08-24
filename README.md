# recommendation-system


## Training
PYTHONPATH="." luigi --module recommendation.task.model.auto_encoder UnconstrainedAutoEncoderTraining --project yelp_business_autoencoder --local-scheduler

## Evaluation

PYTHONPATH="." luigi --module recommendation.task.ifood GenerateReconstructedInteractionMatrix --local-scheduler  --batch-size=2048 --model-module=recommendation.task.model.auto_encoder --model-cls=UnconstrainedAutoEncoderTraining --model-task-id=UnconstrainedAutoEncoderTraining_selu____2048_46f8af394e


PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateMostPopularIfoodModel --local-scheduler --window-filter one_week --model-task-id one_week


PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateRandomIfoodModel --local-scheduler --window-filter one_week --model-task-id one_week
