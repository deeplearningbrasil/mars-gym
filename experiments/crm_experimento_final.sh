var_filter_city='Rio de Janeiro, Brazil'
n_factors=50 
epochs=250 
batch_size=200 
learning_rate=0.0001 
optimizer=adam 
early_stopping_patience=5 
bucket='gs://result-rio-brazil-crm-offpolicy'


#BCE 
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}' --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function bce --epochs 105
#TrivagoLogisticModelTraining_selu____model_11937d2136

# CRM + Model 
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy model --epochs 109
#TrivagoLogisticModelTraining_selu____model_7526476b4b

# CRM + Model 
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy model --epochs 110
#

# CRM + Model + Clip
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --loss-function-params '{"clip": 100}' --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy model --epochs 100

#
# CRM + Model + per_item_in_first_pos
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 2000}'  --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy per_item_in_first_pos --epochs 111
#

## EVAL


# BCE
PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____model_11937d2136" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --policy-estimator-extra-params '{"epochs": 500}' --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"

PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____model_9ba6069be7" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --policy-estimator-extra-params '{"epochs": 500}' --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"

# CRM
PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____model_7526476b4b" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --policy-estimator-extra-params '{"epochs": 500}' --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"



PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____model_00ae2a5fc4" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_item_in_first_pos --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --policy-estimator-extra-params '{"epochs": 500}' --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"
