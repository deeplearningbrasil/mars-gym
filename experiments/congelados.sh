PYTHONPATH="." luigi --module recommendation.task.interaction InteractionTraining --filter-dish "Congelados" --optimizer=adam --learning-rate=0.001 --epochs 500 --use-normalize --use-buys-visits --binary --predictor=simple_logistic_regression --item-embeddings --context-embeddings --use-numerical-content --user-embeddings --n-factors=10  --obs-batch-size 100 --batch-size 100 --num-episodes 50 --full-refit --bandit-policy random --local-scheduler --generator-workers 0 

PYTHONPATH="." nohup luigi --module recommendation.task.interaction InteractionTraining --filter-dish "Congelados" --optimizer=adam --learning-rate=0.001 --epochs 500 --use-normalize --use-buys-visits --binary --predictor=logistic_regression --item-embeddings --context-embeddings --use-numerical-content --user-embeddings --n-factors=10  --obs-batch-size 100 --batch-size 100 --num-episodes 50 --full-refit --bandit-policy random --local-scheduler --generator-workers 0 > nohup1.0 &

PYTHONPATH="." nohup luigi --module recommendation.task.interaction InteractionTraining --filter-dish "Congelados" --optimizer=adam --learning-rate=0.001 --epochs 500 --use-normalize --use-buys-visits --binary --predictor=logistic_regression --item-embeddings --context-embeddings --use-numerical-content --user-embeddings --n-factors=10  --obs-batch-size 100 --batch-size 100 --num-episodes 50 --full-refit --bandit-policy model --local-scheduler --generator-workers 0 > nohup2.0 &

PYTHONPATH="." nohup luigi --module recommendation.task.interaction InteractionTraining --filter-dish "Congelados" --optimizer=adam --learning-rate=0.001 --epochs 500 --use-normalize --use-buys-visits --binary --predictor=logistic_regression --item-embeddings --context-embeddings --use-numerical-content --user-embeddings --n-factors=10  --obs-batch-size 100 --batch-size 100 --num-episodes 50 --full-refit --bandit-policy epsilon_greedy  --bandit-policy-params '{"epsilon": 0.1}' --local-scheduler --generator-workers 0 > nohup3.0 &

PYTHONPATH="." nohup luigi --module recommendation.task.interaction InteractionTraining --filter-dish "Congelados" --optimizer=adam --learning-rate=0.001 --epochs 500 --use-normalize --use-buys-visits --binary --predictor=logistic_regression --item-embeddings --context-embeddings --use-numerical-content --user-embeddings --n-factors=10  --obs-batch-size 100 --batch-size 100 --num-episodes 50 --full-refit --bandit-policy lin_ucb   --local-scheduler --generator-workers 0 > nohup4.0 &

PYTHONPATH="." nohup luigi --module recommendation.task.interaction InteractionTraining --filter-dish "Congelados" --optimizer=adam --learning-rate=0.001 --epochs 500 --use-normalize --use-buys-visits --binary --predictor=logistic_regression --item-embeddings --context-embeddings --use-numerical-content --user-embeddings --n-factors=10  --obs-batch-size 100 --batch-size 100 --num-episodes 50 --full-refit --bandit-policy lin_ts   --local-scheduler --generator-workers 0 > nohup5.0 &

PYTHONPATH="." nohup luigi --module recommendation.task.interaction InteractionTraining --filter-dish "Congelados" --optimizer=adam --learning-rate=0.001 --epochs 500 --use-normalize --use-buys-visits --binary --predictor=logistic_regression --item-embeddings --context-embeddings --use-numerical-content --user-embeddings --n-factors=10  --obs-batch-size 100 --batch-size 100 --num-episodes 50 --full-refit --bandit-policy adaptive   --local-scheduler --generator-workers 0 > nohup6.0 &

PYTHONPATH="." nohup luigi --module recommendation.task.interaction InteractionTraining --filter-dish "Congelados" --optimizer=adam --learning-rate=0.001 --epochs 500 --use-normalize --use-buys-visits --binary --predictor=logistic_regression --item-embeddings --context-embeddings --use-numerical-content --user-embeddings --n-factors=10  --obs-batch-size 100 --batch-size 100 --num-episodes 50 --full-refit --bandit-policy percentile_adaptive  --bandit-policy-params '{"exploration_threshold": 0.7}' --local-scheduler --generator-workers 0 > nohup7.0 &

PYTHONPATH="." nohup luigi --module recommendation.task.interaction InteractionTraining --filter-dish "Congelados" --optimizer=adam --learning-rate=0.001 --epochs 500 --use-normalize --use-buys-visits --binary --predictor=logistic_regression --item-embeddings --context-embeddings --use-numerical-content --user-embeddings --n-factors=10  --obs-batch-size 100 --batch-size 100 --num-episodes 50 --full-refit --bandit-policy explore_then_exploit --bandit-policy-params '{"explore_rounds": 1000}' --local-scheduler --generator-workers 0 > nohup9.0 &

PYTHONPATH="." nohup luigi --module recommendation.task.interaction InteractionTraining --filter-dish "Congelados" --optimizer=adam --learning-rate=0.001 --epochs 500 --use-normalize --use-buys-visits --binary --predictor=logistic_regression --item-embeddings --context-embeddings --use-numerical-content --user-embeddings --n-factors=10  --obs-batch-size 100 --batch-size 100 --num-episodes 50 --full-refit --bandit-policy softmax_explorer  --bandit-policy-params '{"logit_multiplier": 5}' --local-scheduler --generator-workers 0 > nohup10.0 &

PYTHONPATH="." nohup luigi --module recommendation.task.interaction InteractionTraining --filter-dish "Congelados" --optimizer=adam --learning-rate=0.001 --epochs 500 --use-normalize --use-buys-visits --binary --predictor=logistic_regression --item-embeddings --context-embeddings --use-numerical-content --user-embeddings --n-factors=10  --obs-batch-size 100 --batch-size 100 --num-episodes 50 --full-refit --bandit-policy custom_lin_ucb  --local-scheduler --generator-workers 0 > nohup11.0 &