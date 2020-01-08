# !bin/bash

# MostPopular
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateMostPopularIfoodModel --local-scheduler --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateMostPopularIfoodModel --local-scheduler --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

# MostPopularPerUser

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateMostPopularPerUserIfoodModel --local-scheduler --plot-histogram 

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateMostPopularPerUserIfoodModel --local-scheduler --plot-histogram  --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' 

# RAndom

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateRandomIfoodModel --local-scheduler --plot-histogram


PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateRandomIfoodModel --local-scheduler --plot-histogram  --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram


# DirectEstimatorTraining____500____05641ecfa6
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler --model-module=recommendation.task.model.contextual_bandits --model-cls=DirectEstimatorTraining --model-task-id=DirectEstimatorTraining____500____05641ecfa6 --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler --model-module=recommendation.task.model.contextual_bandits --model-cls=DirectEstimatorTraining --model-task-id=DirectEstimatorTraining____500____05641ecfa6 --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler --model-module=recommendation.task.model.contextual_bandits --model-cls=DirectEstimatorTraining --model-task-id=DirectEstimatorTraining____500____05641ecfa6 --bandit-policy model --plot-histogram

# MatrixFactorizationTraining____500_False_40c44eeb5b
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationTraining --model-task-id=MatrixFactorizationTraining____500_False_40c44eeb5b --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationTraining --model-task-id=MatrixFactorizationTraining____500_False_40c44eeb5b --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationTraining --model-task-id=MatrixFactorizationTraining____500_False_40c44eeb5b --bandit-policy model --plot-histogram

# MatrixFactorizationTraining____500_False_4e2a8c17d3
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationTraining --model-task-id=MatrixFactorizationTraining____500_False_4e2a8c17d3 --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationTraining --model-task-id=MatrixFactorizationTraining____500_False_4e2a8c17d3 --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationTraining --model-task-id=MatrixFactorizationTraining____500_False_4e2a8c17d3 --bandit-policy model --plot-histogram

# MatrixFactorizationWithShiftTraining____500____66372fd23d
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationWithShiftTraining --model-task-id=MatrixFactorizationWithShiftTraining____500____66372fd23d --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationWithShiftTraining --model-task-id=MatrixFactorizationWithShiftTraining____500____66372fd23d --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationWithShiftTraining --model-task-id=MatrixFactorizationWithShiftTraining____500____66372fd23d --bandit-policy model --plot-histogram

# MatrixFactorizationWithShiftTraining____500____3ae7f490c8
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationWithShiftTraining --model-task-id=MatrixFactorizationWithShiftTraining____500____3ae7f490c8 --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationWithShiftTraining --model-task-id=MatrixFactorizationWithShiftTraining____500____3ae7f490c8 --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationWithShiftTraining --model-task-id=MatrixFactorizationWithShiftTraining____500____3ae7f490c8 --bandit-policy model --plot-histogram

# UnconstrainedAutoEncoderTraining_selu____500_4471d69030
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateAutoEncoderIfoodModel --model-module=recommendation.task.model.auto_encoder --model-cls=UnconstrainedAutoEncoderTraining --model-task-id=UnconstrainedAutoEncoderTraining_selu____500_4471d69030 --local-scheduler  --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateAutoEncoderIfoodModel --model-module=recommendation.task.model.auto_encoder --model-cls=UnconstrainedAutoEncoderTraining --model-task-id=UnconstrainedAutoEncoderTraining_selu____500_4471d69030 --local-scheduler  --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateAutoEncoderIfoodModel --model-module=recommendation.task.model.auto_encoder --model-cls=UnconstrainedAutoEncoderTraining --model-task-id=UnconstrainedAutoEncoderTraining_selu____500_4471d69030 --local-scheduler  --bandit-policy model --plot-histogram

# VariationalAutoEncoderTraining_selu____500_249fd09ed1
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateAutoEncoderIfoodModel --model-module=recommendation.task.model.auto_encoder --model-cls=VariationalAutoEncoderTraining --model-task-id=VariationalAutoEncoderTraining_selu____500_249fd09ed1 --local-scheduler --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateAutoEncoderIfoodModel --model-module=recommendation.task.model.auto_encoder --model-cls=VariationalAutoEncoderTraining --model-task-id=VariationalAutoEncoderTraining_selu____500_249fd09ed1 --local-scheduler  --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateAutoEncoderIfoodModel --model-module=recommendation.task.model.auto_encoder --model-cls=VariationalAutoEncoderTraining --model-task-id=VariationalAutoEncoderTraining_selu____500_249fd09ed1 --local-scheduler  --bandit-policy model --plot-histogram

# TripletNetTraining____512____8f8a418af8
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetTraining --model-task-id=TripletNetTraining____512____8f8a418af8 --local-scheduler  --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetTraining --model-task-id=TripletNetTraining____512____8f8a418af8 --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.1}' --local-scheduler  --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetTraining --model-task-id=TripletNetTraining____512____8f8a418af8 --bandit-policy model --local-scheduler  --plot-histogram

# TripletNetContentTraining_selu____512_0391c72a86
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetContentTraining --model-task-id=TripletNetContentTraining_selu____512_0391c72a86 --local-scheduler --batch-size 100  --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetContentTraining --model-task-id=TripletNetContentTraining_selu____512_0391c72a86 --local-scheduler --batch-size 100 --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.1}' --local-scheduler  --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetContentTraining --model-task-id=TripletNetContentTraining_selu____512_0391c72a86 --local-scheduler --batch-size 100 --bandit-policy model --local-scheduler  --plot-histogram

# TripletNetSimpleContentTraining_selu____512_4de3fbaa3e
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetSimpleContentTraining --model-task-id=TripletNetSimpleContentTraining_selu____512_4de3fbaa3e --local-scheduler --batch-size 500  --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetSimpleContentTraining --model-task-id=TripletNetSimpleContentTraining_selu____512_4de3fbaa3e --local-scheduler --batch-size 500  --plot-histogram --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.1}' --local-scheduler  

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetSimpleContentTraining --model-task-id=TripletNetSimpleContentTraining_selu____512_4de3fbaa3e --local-scheduler --batch-size 500  --plot-histogram --bandit-policy model --local-scheduler  

# ContextualBanditsTraining_selu____512_1bdce6508b
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_1bdce6508b 

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_1bdce6508b --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_1bdce6508b --bandit-policy model --plot-histogram

# ContextualBanditsTraining_selu____512_54d17b4f72
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_54d17b4f72 

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_54d17b4f72 --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_54d17b4f72 --bandit-policy model --plot-histogram