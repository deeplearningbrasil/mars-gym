# !bin/bash


PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateRandomIfoodModel --local-scheduler 

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateMostPopularIfoodModel --local-scheduler 

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateMostPopularPerUserIfoodModel --local-scheduler 

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationTraining --model-task-id=MatrixFactorizationTraining____500_False_40c44eeb5b

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationTraining --model-task-id=MatrixFactorizationTraining____500_False_4e2a8c17d3

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationWithShiftTraining --model-task-id=MatrixFactorizationWithShiftTraining____500____66372fd23d

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationWithShiftTraining --model-task-id=MatrixFactorizationWithShiftTraining____500____3ae7f490c8

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateAutoEncoderIfoodModel --model-module=recommendation.task.model.auto_encoder --model-cls=UnconstrainedAutoEncoderTraining --model-task-id=UnconstrainedAutoEncoderTraining_selu____500_4471d69030 --local-scheduler 

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateAutoEncoderIfoodModel --model-module=recommendation.task.model.auto_encoder --model-cls=VariationalAutoEncoderTraining --model-task-id=VariationalAutoEncoderTraining_selu____500_249fd09ed1 --local-scheduler 

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetTraining --model-task-id=TripletNetTraining____512____8f8a418af8 --local-scheduler 

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetSimpleContentTraining --model-task-id=TripletNetSimpleContentTraining_selu____512_4de3fbaa3e --local-scheduler --batch-size 500

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_75e7eb41b1 --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.05}'

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_75e7eb41b1 --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}'

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_75e7eb41b1 --bandit-policy random

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_54d17b4f72 --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.05}'

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_54d17b4f72 --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}'

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_54d17b4f72 --bandit-policy random

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_54d17b4f72 --bandit-policy lin_ucb

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_75e7eb41b1 --bandit-policy lin_ucb