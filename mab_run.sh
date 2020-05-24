#!/bin/bash


# PYTHONPATH="." luigi --module recommendation.task.deployment.matrix_factorization PackMatrixFactorization \
# --model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationTraining \
# --model-task-id='MatrixFactorizationTraining____500_False_9fce69a6de' 

# PYTHONPATH="." luigi --module recommendation.task.deployment.auto_encoder PackAutoEncoder --model-cls VariationalAutoEncoderTraining --model-task-id 'VariationalAutoEncoderTraining_selu____500_2d11e525f7' --user-column account_id --user-idx-column account_idx --item-column buys_per_merchant_id --item-idx-column buys_per_merchant_idx --deploy-module cvae

# PYTHONPATH="." luigi --module recommendation.task.deployment.auto_encoder PackAutoEncoder --model-cls UnconstrainedAutoEncoderTraining  --model-task-id 'UnconstrainedAutoEncoderTraining_selu____100_43e5570ed9' --user-column account_id --user-idx-column account_idx --item-column buys_per_merchant_id --item-idx-column buys_per_merchant_idx --deploy-module cdae


# bentoml serve RandomRankingRecommender:latest --port 5000
# bentoml serve MostPopularRankingRecommender:latest --port 5001
# bentoml serve MostPopularPerUserRankingRecommender:latest --port 5002
# bentoml serve MatrixFactorizationRankingRecommender:latest --port 5003
# bentoml serve CVAERankingRecommender:latest --port 5004
# bentoml serve CDAERankingRecommender:latest --port 5005


# outro


# PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5000/rank"]}' --obs-batch-size 1 --val-size 0 --obs "Random"

# PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank"]}' --obs-batch-size 1 --val-size 0 --obs "MostPopular"

# PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5002/rank"]}' --obs-batch-size 1 --val-size 0 --obs "MostPopularPerUser"

# PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5003/rank"]}' --obs-batch-size 1 --val-size 0 --obs "MatrixFactorization"

# PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5004/rank"]}' --obs-batch-size 1 --val-size 0 --obs "CVAE"

# PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote --bandit-policy-params '{"endpoints": ["http://localhost:5005/rank"]}' --obs-batch-size 1 --val-size 0 --obs "CDAE"



for i in $(seq 1 10) 
do

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_contextual_softmax --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank", "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"], "logit_multiplier": 10}' --obs-batch-size 1 --val-size 0 --obs "[Contextual]Softmax(10) $i" --seed $i

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_contextual_softmax --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank", "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"], "logit_multiplier": 50}' --obs-batch-size 1 --val-size 0 --obs "[Contextual]Softmax(50) $i" --seed $i

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_contextual_softmax --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank", "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"], "logit_multiplier": 100}' --obs-batch-size 1 --val-size 0 --obs "[Contextual]Softmax(100) $i" --seed $i

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_contextual_softmax --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank", "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"], "logit_multiplier": 500}' --obs-batch-size 1 --val-size 0 --obs "[Contextual]Softmax(500) $i" --seed $i


PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_contextual_epsilon_greedy --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank",  "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"], "epsilon": 0.05}' --obs-batch-size 1 --val-size 0 --obs "[Contextual]e-greedy(5) $i" --seed $i

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_contextual_epsilon_greedy --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank",  "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"], "epsilon": 0.1}' --obs-batch-size 1 --val-size 0 --obs "[Contextual]e-greedy(10) $i" --seed $i

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_contextual_epsilon_greedy --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank",  "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"], "epsilon": 0.2}' --obs-batch-size 1 --val-size 0 --obs "[Contextual]e-greedy(20) $i" --seed $i


done

for i in $(seq 1 10) 
do

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_epsilon_greedy --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank",  "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"], "window_reward": 500}' --obs-batch-size 1 --val-size 0 --obs "[MAB] e-greedy $i" --seed $i


PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_ucb --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank",  "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"]}' --obs-batch-size 1 --val-size 0 --obs "[MAB] UCB  $i" --seed $i


PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_contextual_epsilon_greedy --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank",  "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"]}' --obs-batch-size 1 --val-size 0 --obs "[Contextual] e-greedy $i" --seed $i

PYTHONPATH="." luigi --module recommendation.task.model.ensamble_mab EnsambleMABInteraction --project ifood_ensamble_mab --bandit-policy remote_contextual_softmax --bandit-policy-params '{"endpoints": ["http://localhost:5001/rank", "http://localhost:5003/rank", "http://localhost:5004/rank", "http://localhost:5005/rank"]}' --obs-batch-size 1 --val-size 0 --obs "[Contextual] Softmax $i" --seed $i

done


