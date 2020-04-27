#!/bin/bash

var_filter_city='recsys'
num_episodes=1 
n_factors=50 
epochs=250 
obs_batch_size=3000
batch_size=200 
learning_rate=0.001 
optimizer=adam 
val_split_type=random 
early_stopping_patience=5 
bucket='gs://result-recsys-cities-metrics'



## softmax_explorer
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit_bce --data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  --output-model-dir $bucket --test-size 0.2 --loss-function bce

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  --output-model-dir $bucket --test-size 0.2 


# Model
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit_bce --data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy model  --output-model-dir $bucket --test-size 0.2 --loss-function bce


PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy model  --output-model-dir $bucket --test-size 0.2


# lin_ucb
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1e-5}'    --output-model-dir $bucket --test-size 0.2

# custom_lin_ucb
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1e-5}'   --output-model-dir $bucket --test-size 0.2


# Epsilon Greedy
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.1}'  --output-model-dir $bucket --test-size 0.2

# Adaptive
#https://www.wolframalpha.com/input/?i=0.1%3D0.7%281-r%29%5E20000
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.7, "decay_rate": 0.0000972907743983833}'   --output-model-dir $bucket --test-size 0.2

# Percentil Adaptive
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy percentile_adaptive --bandit-policy-params '{"exploration_threshold": 0.7}'   --output-model-dir $bucket --test-size 0.2


## Explore the Exploit
# #https://www.wolframalpha.com/input/?i=0.1%3D0.8%281-r%29%5E2000
#
#
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy explore_then_exploit --bandit-policy-params '{"explore_rounds": 1000, "decay_rate": 0.0001872157}'   --output-model-dir $bucket --test-size 0.2

# Lin_ts
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy lin_ts --bandit-policy-params '{"v_sq": 0.1}'  --output-model-dir $bucket --test-size 0.2


# Random
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project test_fixed_trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy random  --output-model-dir $bucket --test-size 0.2

# First Item
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project test_fixed_trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy fixed --bandit-policy-params '{"arg": 1}' --observation "First Item"  --output-model-dir $bucket --test-size 0.2


# Popular Item
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project test_fixed_trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy fixed --bandit-policy-params '{"arg": 2}' --observation "Popular Item"  --output-model-dir $bucket --test-size 0.2
