
# Epsilon Greedy

PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 100 --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.05}' > nohup3.1 &

PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 100 --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.1}' > nohup3.2 &

PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 100 --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.2}' > nohup3.3 &

# lin_ucb

PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 100 --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1e-5}' --generator-workers 4 > nohup4.1 &

PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 100 --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1e-2}' --generator-workers 4 > nohup4.2 &

PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 100 --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1}' --generator-workers 4 > nohup4.3 &

# custom_lin_ucb

PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 100 --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1e-5}' --generator-workers 4 > nohup5 &

PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 100 --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1e-2}' --generator-workers 4 > nohup5 &

PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 100 --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1}' --generator-workers 4 > nohup5 &

## Lin TS

# PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 300 --bandit-policy lin_ts > nohup6 &


## softmax_explorer

PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 100 --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 0.5}'  > nohup7 &

PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 100 --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 1}'  > nohup7 &

PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 100 --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  > nohup7 &

## Percentile_adaptive

PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 300 --bandit-policy percentile_adaptive --bandit-policy-params '{"exploration_threshold": 0.2}' > nohup9 &

PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 300 --bandit-policy percentile_adaptive --bandit-policy-params '{"exploration_threshold": 0.5}' > nohup9 &

PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 300 --bandit-policy percentile_adaptive --bandit-policy-params '{"exploration_threshold": 0.7}' > nohup9 &

## 
PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 300 --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.7, "decay_rate": 0.000199566512577}' > nohup8 & 
#https://www.wolframalpha.com/input/?i=0.1%3D0.8%281-r%29%5E2000

PYTHONPATH="." nohup  luigi --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 300 --bandit-policy explore_then_exploit --bandit-policy-params '{"explore_rounds": 1000, "decay_rate": 0.0001872157}' > nohup10 &
