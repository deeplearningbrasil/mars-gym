######

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
bucket='gs://result-rio-brazil-crm-test-final'

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --loss-function bce --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy model --seed 51 --output-model-dir $bucket --test-size 0.2
# TrivagoLogisticModelInteraction_selu____model_8d1689a36c



# SEM CRM
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --loss-function bce --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  --seed 51 --output-model-dir $bucket --test-size 0.2
#TrivagoLogisticModelInteraction_selu____softmax_explorer_9d575d77b8


#  COM CRM
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --loss-function crm  --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  --seed 51 --output-model-dir $bucket --test-size 0.2
# TrivagoLogisticModelInteraction_selu____softmax_explorer_fd1e2d488f


#  COM CRM + CLIP 2
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --loss-function crm --loss-function-params '{"clip": 2}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  --seed 51 --output-model-dir $bucket --test-size 0.2
# TrivagoLogisticModelInteraction_selu____softmax_explorer_13be3587d9


#  COM CRM + CLIP 15
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --loss-function crm --loss-function-params '{"clip": 15}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  --seed 51 --output-model-dir $bucket --test-size 0.2
#TrivagoLogisticModelInteraction_selu____softmax_explorer_f6ebb8c343

#  COM CRM + CLIP 50
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --loss-function crm --loss-function-params '{"clip": 50}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  --seed 51 --output-model-dir $bucket --test-size 0.2
#TrivagoLogisticModelInteraction_selu____softmax_explorer_4fb21a59b8

##################### Evaluation


PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id 'TrivagoLogisticModelInteraction_selu____model_8d1689a36c' --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --policy-estimator-extra-params '{"epochs": 200}' --eval-cips-cap 15
# EvaluateTrivagoTestSetPredictions_500_TrivagoLogisticM_100_d2b1723d5a


PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id 'TrivagoLogisticModelInteraction_selu____softmax_explorer_9d575d77b8' --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --policy-estimator-extra-params '{"epochs": 200}' --eval-cips-cap 15
# EvaluateTrivagoTestSetPredictions_500_TrivagoLogisticM_100_49994e8efc

PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id 'TrivagoLogisticModelInteraction_selu____softmax_explorer_fd1e2d488f' --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --policy-estimator-extra-params '{"epochs": 200}' --eval-cips-cap 15
# EvaluateTrivagoTestSetPredictions_500_TrivagoLogisticM_100_21548db875

PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id 'TrivagoLogisticModelInteraction_selu____softmax_explorer_13be3587d9' --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --policy-estimator-extra-params '{"epochs": 200}' --eval-cips-cap 15
# EvaluateTrivagoTestSetPredictions_500_TrivagoLogisticM_100_ad1aa9146a


PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id 'TrivagoLogisticModelInteraction_selu____softmax_explorer_f6ebb8c343' --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --policy-estimator-extra-params '{"epochs": 200}' --eval-cips-cap 15
# EvaluateTrivagoTestSetPredictions_500_TrivagoLogisticM_100_5e2b7cb20f

PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelInteraction --model-task-id 'TrivagoLogisticModelInteraction_selu____softmax_explorer_4fb21a59b8' --fairness-columns "[\"device_idx\"]" --local-scheduler --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --policy-estimator-extra-params '{"epochs": 200}' --eval-cips-cap 15
# EvaluateTrivagoTestSetPredictions_500_TrivagoLogisticM_100_8889670d3c





















##############################################################################################
 

#  COM CRM
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --loss-function crm  --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  --seed 51 --output-model-dir $bucket --test-size 0.2 --crm-ps-strategy "dataset"
# 

#  COM CRM + CLIP 2
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --loss-function crm --loss-function-params '{"clip": 2}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  --seed 51 --output-model-dir $bucket --test-size 0.2 --crm-ps-strategy "dataset"
# 

#  COM CRM + CLIP 15
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --loss-function crm --loss-function-params '{"clip": 15}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  --seed 51 --output-model-dir $bucket --test-size 0.2 --crm-ps-strategy "dataset"
#

#  COM CRM + CLIP 50
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --loss-function crm --loss-function-params '{"clip": 50}' --metrics '["loss"]' --epochs $epochs --obs-batch-size $obs_batch_size --early-stopping-patience $early_stopping_patience --batch-size $batch_size --num-episodes $num_episodes --val-split-type $val_split_type --full-refit --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  --seed 51 --output-model-dir $bucket --test-size 0.2 --crm-ps-strategy "dataset"
#
