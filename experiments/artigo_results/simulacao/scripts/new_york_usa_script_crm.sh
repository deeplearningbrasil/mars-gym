#!/bin/bash

var_filter_city='New York, USA'
num_episodes=1 
n_factors=50 
epochs=250 
obs_batch_size=1000
batch_size=200 
learning_rate=0.001 
optimizer=adam 
val_split_type=random 
early_stopping_patience=5 
bucket='gs://result-new-york-usa-crm'


# Random
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project test_fixed_trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "New York, USA", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy random --output-model-dir $bucket

# First Item
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project test_fixed_trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "New York, USA", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy fixed --bandit-policy-params '{"arg": 1}' --observation "First Item" --output-model-dir $bucket


# Popular Item
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project test_fixed_trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "New York, USA", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy fixed --bandit-policy-params '{"arg": 2}' --observation "Popular Item" --output-model-dir $bucket

for i in $(seq 1 10) 
do

# Model
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "New York, USA", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy model  --seed $i --output-model-dir $bucket



# lin_ucb
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "New York, USA", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1e-5}'    --seed $i --output-model-dir $bucket

# custom_lin_ucb
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "New York, USA", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1e-5}'   --seed $i --output-model-dir $bucket

# Lin_ts
#PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "New York, USA", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy lin_ts --bandit-policy-params '{"v_sq": 0.1}'  --seed $i --output-model-dir $bucket

# Epsilon Greedy
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "New York, USA", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.1}'  --seed $i --output-model-dir $bucket

# Adaptive
#https://www.wolframalpha.com/input/?i=0.1%3D0.7%281-r%29%5E20000
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "New York, USA", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.7, "decay_rate": 0.0000972907743983833}'   --seed $i --output-model-dir $bucket

# Percentil Adaptive
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "New York, USA", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy percentile_adaptive --bandit-policy-params '{"exploration_threshold": 0.7}'   --seed $i --output-model-dir $bucket


## softmax_explorer
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "New York, USA", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  --seed $i --output-model-dir $bucket


## Explore the Exploit
# #https://www.wolframalpha.com/input/?i=0.1%3D0.8%281-r%29%5E2000
#
#
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "New York, USA", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy explore_then_exploit --bandit-policy-params '{"explore_rounds": 1000, "decay_rate": 0.0001872157}'   --seed $i --output-model-dir $bucket


done
