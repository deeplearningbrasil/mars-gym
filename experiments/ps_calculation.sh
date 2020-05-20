PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id TrivagoLogisticModelInteraction_selu____softmax_explorer_acf090f664 --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining

PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id TrivagoLogisticModelInteraction_selu____softmax_explorer_acf090f664 --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_pos_item_idx

PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id TrivagoLogisticModelInteraction_selu____softmax_explorer_acf090f664 --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_item

PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id TrivagoLogisticModelInteraction_selu____softmax_explorer_acf090f664 --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_item_in_first_pos

PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id TrivagoLogisticModelInteraction_selu____softmax_explorer_acf090f664 --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_item_given_pos

PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id TrivagoLogisticModelInteraction_selu____softmax_explorer_acf090f664 --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_logistic_regression_of_pos_item_idx_and_item

PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id TrivagoLogisticModelInteraction_selu____softmax_explorer_acf090f664 --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_logistic_regression_of_pos_item_idx_and_item_ps


#----------------------------


var_filter_city='Rio de Janeiro, Brazil'
num_episodes=1 
n_factors=50 
epochs=250 
obs_batch_size=1000
batch_size=200 
learning_rate=0.001 
optimizer=adam 
val_split_type=random 
early_stopping_patience=5 
bucket='gs://result-test-ifood'

PYTHONPATH="." nohup luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  --output-model-dir $bucket --test-size 0.2 --loss-function bce --seed 45 > nohup_1 &


PYTHONPATH="." nohup luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  --output-model-dir $bucket --test-size 0.2 --loss-function crm --seed 45 > nohup_2 &

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
bucket='gs://result-test-ifood'


## softmax_explorer
PYTHONPATH="." nohup luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  --output-model-dir $bucket --test-size 0.2 --loss-function bce > nohup_1 &

PYTHONPATH="." nohup luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "recsys", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 1}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  --output-model-dir $bucket --test-size 0.2  > nohup_2 &


# eval

PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id TrivagoLogisticModelInteraction_selu____softmax_explorer_1b974734d3 --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_pos_item_idx


PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id TrivagoLogisticModelInteraction_selu____softmax_explorer_9bf11a9c06 --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_pos_item_idx


PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id TrivagoLogisticModelInteraction_selu____softmax_explorer_583dbdd50c --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_pos_item_idx


PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id TrivagoLogisticModelInteraction_selu____softmax_explorer_ddc05422ae --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_pos_item_idx

