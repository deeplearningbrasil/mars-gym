#bin/bash

GCS="gs://deepfood-results-sample"

#Geral
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Lausanne, Switzerland"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy random

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy random

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Boston, USA"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy random

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Vancouver, Canada"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy random

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy random

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Copenhagen, Denmark"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy random

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy random

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Hong Kong, Hong Kong"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy random

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Vienna, Austria"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy random


PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Barcelona, Spain"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy random


PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy random


PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Barcelona, Spain"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy random

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "New York, USA"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy random

ss
# Epsilon Greedy
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.05}' 

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.1}'  

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.2}'  

# lin_ucb
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1e-5}'   

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1e-2}'   

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1e-1}'   

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1}'   

# custom_lin_ucb
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1e-5}'  

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1e-2}'  

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1e-1}'  

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1}'  

## Lin TS
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy lin_ts --bandit-policy-params '{"v_sq": 0.1}' 

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy lin_ts --bandit-policy-params '{"v_sq": 0.5}' 

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy lin_ts --bandit-policy-params '{"v_sq": 1}' 


## softmax_explorer
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 0.5}' 

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 1}'  

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  

## Percentile_adaptive
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy percentile_adaptive --bandit-policy-params '{"exploration_threshold": 0.2}'  

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy percentile_adaptive --bandit-policy-params '{"exploration_threshold": 0.5}'  

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy percentile_adaptive --bandit-policy-params '{"exploration_threshold": 0.7}'  


## Adaptative
## 
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.7, "decay_rate": 0.000199566512577}'  

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.5, "decay_rate": 0.000199566512577}'  

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.3, "decay_rate": 0.000199566512577}'  

#https://www.wolframalpha.com/input/?i=0.1%3D0.7%281-r%29%5E20000
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.7, "decay_rate": 0.0000972907743983833}'  

#https://www.wolframalpha.com/input/?i=0.1%3D0.5%281-r%29%5E20000
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.5, "decay_rate": 0.0000804686578455631}'  

#https://www.wolframalpha.com/input/?i=0.1%3D0.3%281-r%29%5E20000
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.3, "decay_rate": 0.0000549291057748284}'  


## Explore the Exploit
# #https://www.wolframalpha.com/input/?i=0.1%3D0.8%281-r%29%5E2000
#
#
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy explore_then_exploit --bandit-policy-params '{"explore_rounds": 1000, "decay_rate": 0.0001872157}'  

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_models TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 1 --obs-batch-size 20000 --early-stopping-patience 10 --batch-size 200 --num-episodes 1 --output-model-dir $GCS --bandit-policy explore_then_exploit --bandit-policy-params '{"explore_rounds": 1000, "decay_rate": 0.000115122627531392}'  