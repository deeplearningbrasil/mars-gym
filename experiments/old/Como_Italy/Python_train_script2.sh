# Model
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy", "window_hist":10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 500 --early-stopping-patience 5 --batch-size 200 --loss-function bce --num-episodes 100 --val-split-type random --full-refit --bandit-policy model

# Random
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project test_fixed_trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy", "window_hist":10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 500 --early-stopping-patience 5 --batch-size 200 --loss-function bce --num-episodes 100 --val-split-type random --full-refit --bandit-policy random

# First Item
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project test_fixed_trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy", "window_hist":10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 500 --early-stopping-patience 5 --batch-size 200 --loss-function bce --num-episodes 100 --val-split-type random --full-refit --bandit-policy fixed --bandit-policy-params '{"arg": 1}' --observation "First Item"

# Popular Item
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project test_fixed_trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy", "window_hist":10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 500 --early-stopping-patience 5 --batch-size 200 --loss-function bce --num-episodes 100 --val-split-type random --full-refit --bandit-policy fixed --bandit-policy-params '{"arg": 2}' --observation "Popular Item"

# lin_ucb
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy", "window_hist":10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 500 --early-stopping-patience 5 --batch-size 200 --loss-function bce --num-episodes 100 --val-split-type random --full-refit --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 0.01}'   

# custom_lin_ucb
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy", "window_hist":10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 500 --early-stopping-patience 5 --batch-size 200 --loss-function bce --num-episodes 100 --val-split-type random --full-refit --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 0.01}'  

# Lin_ts
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy", "window_hist":10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 500 --early-stopping-patience 5 --batch-size 200 --loss-function bce --num-episodes 100 --val-split-type random --full-refit --bandit-policy lin_ts --bandit-policy-params '{"v_sq": 0.1}' 

# Epsilon Greedy
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy", "window_hist":10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 500 --early-stopping-patience 5 --batch-size 200 --loss-function bce --num-episodes 100 --val-split-type random --full-refit --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.1}' 

# Adaptive
#https://www.wolframalpha.com/input/?i=0.1%3D0.7%281-r%29%5E20000
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy", "window_hist":10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 500 --early-stopping-patience 5 --batch-size 200 --loss-function bce --num-episodes 100 --val-split-type random --full-refit --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.7, "decay_rate": 0.0000972907743983833}'  

# Percentil Adaptive
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy", "window_hist":10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 500 --early-stopping-patience 5 --batch-size 200 --loss-function bce --num-episodes 100 --val-split-type random --full-refit --bandit-policy percentile_adaptive --bandit-policy-params '{"exploration_threshold": 0.7}'  


## softmax_explorer
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy", "window_hist":10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 500 --early-stopping-patience 5 --batch-size 200 --loss-function bce --num-episodes 100 --val-split-type random --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}' 


## Explore the Exploit
# #https://www.wolframalpha.com/input/?i=0.1%3D0.8%281-r%29%5E2000
#
#
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy", "window_hist":10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 500 --early-stopping-patience 5 --batch-size 200 --loss-function bce --num-episodes 100 --val-split-type random --full-refit --bandit-policy explore_then_exploit --bandit-policy-params '{"explore_rounds": 1000, "decay_rate": 0.0001872157}'  
