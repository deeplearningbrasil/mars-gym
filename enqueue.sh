# !bin/bash

# MostPopular
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateMostPopularIfoodModel --local-scheduler --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateMostPopularIfoodModel --local-scheduler --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateMostPopularIfoodModel --local-scheduler --plot-histogram --bandit-policy model

# MostPopularPerUser

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateMostPopularPerUserIfoodModel --local-scheduler --plot-histogram 

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateMostPopularPerUserIfoodModel --local-scheduler --plot-histogram  --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' 

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateMostPopularPerUserIfoodModel --local-scheduler --plot-histogram --bandit-policy model

# RAndom

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateRandomIfoodModel --local-scheduler --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateRandomIfoodModel --local-scheduler --plot-histogram  --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateRandomIfoodModel --local-scheduler --plot-histogram --bandit-policy model

# DirectEstimatorTraining____500____591fc2ad60
PYTHONPATH="." luigi --module recommendation.task.model.contextual_bandits DirectEstimatorTraining --project ifood_offpolicy_direct_estimator  --local-scheduler

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler --model-module=recommendation.task.model.contextual_bandits --model-cls=DirectEstimatorTraining --model-task-id=DirectEstimatorTraining____500____591fc2ad60 --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler --model-module=recommendation.task.model.contextual_bandits --model-cls=DirectEstimatorTraining --model-task-id=DirectEstimatorTraining____500____591fc2ad60 --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler --model-module=recommendation.task.model.contextual_bandits --model-cls=DirectEstimatorTraining --model-task-id=DirectEstimatorTraining____500____591fc2ad60 --bandit-policy model --plot-histogram

# MatrixFactorizationTraining____500_False_6fa6e5fea5
PYTHONPATH="." luigi --module recommendation.task.model.matrix_factorization MatrixFactorizationTraining --project ifood_binary_buys_cf --n-factors 100 --binary --loss-function bce --metrics '["loss", "acc"]' --epochs 300 --local-scheduler

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationTraining --model-task-id=MatrixFactorizationTraining____500_False_6fa6e5fea5 --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationTraining --model-task-id=MatrixFactorizationTraining____500_False_6fa6e5fea5 --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationTraining --model-task-id=MatrixFactorizationTraining____500_False_6fa6e5fea5 --bandit-policy model --plot-histogram

# MatrixFactorizationTraining____500_False_c9dcbcf1c1
PYTHONPATH="." luigi --module recommendation.task.model.matrix_factorization MatrixFactorizationTraining --project ifood_binary_buys_cf_with_random_negative --n-factors 100 --binary --loss-function bce --metrics '["loss", "acc"]' --epochs 300 --local-scheduler

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationTraining --model-task-id=MatrixFactorizationTraining____500_False_c9dcbcf1c1 --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationTraining --model-task-id=MatrixFactorizationTraining____500_False_c9dcbcf1c1 --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationTraining --model-task-id=MatrixFactorizationTraining____500_False_c9dcbcf1c1 --bandit-policy model --plot-histogram

# MatrixFactorizationWithShiftTraining____500____0b8e640cb9
PYTHONPATH="." luigi --module recommendation.task.model.matrix_factorization MatrixFactorizationWithShiftTraining --project ifood_binary_buys_with_shift_cf --n-factors 100 --metrics '["loss", "acc"]' --loss-function bce --optimizer-params '{"weight_decay": 1e-8}' --epochs 300 --local-scheduler

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationWithShiftTraining --model-task-id=MatrixFactorizationWithShiftTraining____500____0b8e640cb9 --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationWithShiftTraining --model-task-id=MatrixFactorizationWithShiftTraining____500____0b8e640cb9 --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationWithShiftTraining --model-task-id=MatrixFactorizationWithShiftTraining____500____0b8e640cb9 --bandit-policy model --plot-histogram

# MatrixFactorizationWithShiftTraining____500____c648491c28
PYTHONPATH="." luigi --module recommendation.task.model.matrix_factorization MatrixFactorizationWithShiftTraining --project ifood_binary_buys_with_shift_cf_with_random_negative --n-factors 100 --metrics '["loss", "acc"]' --loss-function bce --optimizer-params '{"weight_decay": 1e-8}' --epochs 300 --local-scheduler

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationWithShiftTraining --model-task-id=MatrixFactorizationWithShiftTraining____500____c648491c28 --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationWithShiftTraining --model-task-id=MatrixFactorizationWithShiftTraining____500____c648491c28 --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationWithShiftTraining --model-task-id=MatrixFactorizationWithShiftTraining____500____c648491c28 --bandit-policy model --plot-histogram

# UnconstrainedAutoEncoderTraining_selu____500_c18722c4e2
PYTHONPATH="." luigi --module recommendation.task.model.auto_encoder UnconstrainedAutoEncoderTraining --project ifood_user_cdae --binary --optimizer adam --learning-rate 1e-4 --batch-size 500 --generator-workers 0 --loss-function focal --loss-function-params '{"gamma": 10.0, "alpha": 1664.0}' --loss-wrapper none --data-frames-preparation-extra-params '{"split_per_user": "True"}' --gradient-norm-clipping 2.0 --gradient-norm-clipping-type 1 --data-transformation support_based --epochs 300 --local-scheduler

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateAutoEncoderIfoodModel --model-module=recommendation.task.model.auto_encoder --model-cls=UnconstrainedAutoEncoderTraining --model-task-id=UnconstrainedAutoEncoderTraining_selu____500_c18722c4e2 --local-scheduler  --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateAutoEncoderIfoodModel --model-module=recommendation.task.model.auto_encoder --model-cls=UnconstrainedAutoEncoderTraining --model-task-id=UnconstrainedAutoEncoderTraining_selu____500_c18722c4e2 --local-scheduler  --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateAutoEncoderIfoodModel --model-module=recommendation.task.model.auto_encoder --model-cls=UnconstrainedAutoEncoderTraining --model-task-id=UnconstrainedAutoEncoderTraining_selu____500_c18722c4e2 --local-scheduler  --bandit-policy model --plot-histogram

# VariationalAutoEncoderTraining_selu____500_fc62ac744a
PYTHONPATH="." luigi --module recommendation.task.model.auto_encoder VariationalAutoEncoderTraining --project ifood_user_cdae --generator-workers=0 --local-scheduler --batch-size=500 --optimizer=adam --lr-scheduler=step --lr-scheduler-params='{"step_size": 5, "gamma": 0.8}'  --activation-function=selu --encoder-layers=[600,200] --decoder-layers=[600] --loss-function=vae_loss --loss-function-params='{"anneal": 0.01}'  --data-transformation=support_based --data-frames-preparation-extra-params='{"split_per_user": "True"}' --epochs=300 --learning-rate=0.001 

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateAutoEncoderIfoodModel --model-module=recommendation.task.model.auto_encoder --model-cls=VariationalAutoEncoderTraining --model-task-id=VariationalAutoEncoderTraining_selu____500_fc62ac744a --local-scheduler --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateAutoEncoderIfoodModel --model-module=recommendation.task.model.auto_encoder --model-cls=VariationalAutoEncoderTraining --model-task-id=VariationalAutoEncoderTraining_selu____500_fc62ac744a --local-scheduler  --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateAutoEncoderIfoodModel --model-module=recommendation.task.model.auto_encoder --model-cls=VariationalAutoEncoderTraining --model-task-id=VariationalAutoEncoderTraining_selu____500_fc62ac744a --local-scheduler  --bandit-policy model --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.iterator_eval.base IterationEvaluationTask --local-scheduler --model-task-id=VariationalAutoEncoderTraining_selu____500_fc62ac744a --model-module=recommendation.task.model.auto_encoder --model-cls=VariationalAutoEncoderTraining --model-module-eval=recommendation.task.ifood   --model-cls-eval=EvaluateAutoEncoderIfoodModel --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}'


# TripletNetTraining____512____19e56afb96
PYTHONPATH="." luigi --module recommendation.task.model.triplet_net TripletNetTraining --project ifood_binary_buys_triplet_with_random_negative  --batch-size=512 --optimizer=adam --lr-scheduler=step --lr-scheduler-params='{"step_size": 5, "gamma": 0.8123}' --learning-rate=0.001 --n-factors=100 --loss-function=weighted_triplet --loss-function-params='{"balance_factor": 2500.0}' --local-scheduler --epochs=300 

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetTraining --model-task-id=TripletNetTraining____512____19e56afb96 --local-scheduler  --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetTraining --model-task-id=TripletNetTraining____512____19e56afb96 --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.1}' --local-scheduler  --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetTraining --model-task-id=TripletNetTraining____512____19e56afb96 --bandit-policy model --local-scheduler  --plot-histogram

# TripletNetContentTraining_selu____512_0c175685c0
PYTHONPATH="." luigi --module recommendation.task.model.triplet_net TripletNetContentTraining --project ifood_binary_buys_content_triplet_with_random_negative  --batch-size=512 --optimizer=adam --lr-scheduler=step --lr-scheduler-params='{"step_size": 5, "gamma": 0.8123}' --learning-rate=0.0001 --epochs=300 --n-factors=100 --loss-function=weighted_triplet --loss-function-params='{"balance_factor": 0.9}' --content-layers="[64,10]" --local-scheduler

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetContentTraining --model-task-id=TripletNetContentTraining_selu____512_0c175685c0 --local-scheduler --batch-size 100  --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetContentTraining --model-task-id=TripletNetContentTraining_selu____512_0c175685c0 --local-scheduler --batch-size 100 --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.1}' --local-scheduler  --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetContentTraining --model-task-id=TripletNetContentTraining_selu____512_0c175685c0 --local-scheduler --batch-size 100 --bandit-policy model --local-scheduler  --plot-histogram

# TripletNetSimpleContentTraining_selu____512_19dad0ffa3
PYTHONPATH="." luigi --module recommendation.task.model.triplet_net TripletNetSimpleContentTraining --project ifood_binary_buys_content_triplet_with_random_negative  --batch-size=512 --optimizer=adam --lr-scheduler=step --lr-scheduler-params='{"step_size": 5, "gamma": 0.8123}' --learning-rate=0.0001 --epochs=300 --n-factors=100 --loss-function=weighted_triplet --loss-function-params='{"balance_factor": 0.9}' --content-layers="[64,10]" --local-scheduler

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetSimpleContentTraining --model-task-id=TripletNetSimpleContentTraining_selu____512_19dad0ffa3 --local-scheduler --batch-size 500  --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetSimpleContentTraining --model-task-id=TripletNetSimpleContentTraining_selu____512_19dad0ffa3 --local-scheduler --batch-size 500  --plot-histogram --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.1}' --local-scheduler  

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetSimpleContentTraining --model-task-id=TripletNetSimpleContentTraining_selu____512_19dad0ffa3 --local-scheduler --batch-size 500  --plot-histogram --bandit-policy model --local-scheduler  

# ContextualBanditsTraining_selu____512_1ef3c7b713
PYTHONPATH="." luigi --module recommendation.task.model.contextual_bandits ContextualBanditsTraining --project ifood_contextual_bandit --local-scheduler --batch-size=512 --optimizer=radam --lr-scheduler=step --lr-scheduler-params='{"step_size": 5, "gamma": 0.801}' --learning-rate=0.001 --loss-function=crm  --use-normalize --use-buys-visits  --content-layers=[256,128,64]  --binary --predictor=logistic_regression --context-embeddings --use-numerical-content --user-embeddings --n-factors=100 --epochs 300

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_1ef3c7b713 

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_1ef3c7b713 --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_1ef3c7b713 --bandit-policy model --plot-histogram

# ContextualBanditsTraining_selu____512_b7903a39eb
PYTHONPATH="." luigi --module recommendation.task.model.contextual_bandits ContextualBanditsTraining --project ifood_contextual_bandit --local-scheduler --batch-size=512 --optimizer=radam --lr-scheduler=step --lr-scheduler-params='{"step_size": 5, "gamma": 0.801}' --learning-rate=0.001 --loss-function=crm  --use-normalize --use-buys-visits  --content-layers=[256,128,64]  --binary --predictor=factorization_machine --context-embeddings --use-numerical-content --user-embeddings --n-factors=100 --epochs 300

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_b7903a39eb 

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_b7903a39eb --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --plot-histogram

PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_b7903a39eb --bandit-policy model --plot-histogram


DATASET_PROCESSED_PATH="./output/ifood/dataset_5" PYTHONPATH="." nohup luigi --module recommendation.task.iterator_eval.iteraction_evaluation IterationEvaluationTask --local-scheduler --model-task-id=ContextualBanditsTraining_selu____512_0771f4fe24 --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-module-eval=recommendation.task.ifood   --model-cls-eval=EvaluateIfoodFullContentModel --run-type=reinforcement --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --batch-size 55000 > nohup_dt5 &