var_filter_city='Rio de Janeiro, Brazil'
n_factors=50 
epochs=250 
batch_size=200 
learning_rate=0.0001 
optimizer=adam 
early_stopping_patience=5 
bucket='gs://result-rio-brazil-crm-offpolicy'


#BCE 
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}' --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function bce --epochs 101
#TrivagoLogisticModelTraining_selu____200_f036a0f1fa

# CRM + Model 
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy model --epochs 101
#TrivagoLogisticModelTraining_selu____200_631417f217

# CRM + Model + Clip
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --loss-function-params '{"clip": 100}' --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy model --policy-estimator-extra-params '{"epochs": 500}' --epochs 100
#TrivagoLogisticModelTraining_selu____200_0f933707b1

# CRM + Model + Clip
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --loss-function-params '{"clip": 1000}' --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy per_probability --policy-estimator-extra-params '{"epochs": 500}' --epochs 101


## EVAL

# BCE
PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoModelTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____200_f036a0f1fa" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --policy-estimator-extra-params '{"epochs": 500}' --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"


# CRM + Model + Clip
PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoModelTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____200_c5c7d72ea0" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --policy-estimator-extra-params '{"epochs": 500}' --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"




# PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --loss-function-params '{"clip": 15}' --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy model --policy-estimator-extra-params '{"epochs": 1}' --epochs 1

PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --loss-function-params '{"clip": 15}' --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy model --policy-estimator-extra-params '{"epochs": 500}' --epochs 251

# crm + dummy
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --loss-function-params '{"clip": 15}' --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy dummy --policy-estimator-extra-params '{"epochs": 1000}' --epochs 251

# crm + model
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --loss-function-params '{"clip": 15}' --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy model --policy-estimator-extra-params '{"epochs": 1000}' --epochs 251
#TrivagoLogisticModelTraining_selu____200_6e8e411c36

# crm + prob
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --loss-function-params '{"clip": 15}' --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy per_probability --policy-estimator-extra-params '{"epochs": 1000}' --epochs 251
#TrivagoLogisticModelTraining_selu____200_c8ff8d87b5

# crm per_pos_item_idx
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --loss-function-params '{"clip": 15}' --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy per_pos_item_idx --policy-estimator-extra-params '{"epochs": 1000}' --epochs 251
#TrivagoLogisticModelTraining_selu____200_3d96e1a1d3

# crm per_logistic_regression_of_pos_item_idx_and_item_ps
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --loss-function-params '{"clip": 15}' --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy per_logistic_regression_of_pos_item_idx_and_item_ps --policy-estimator-extra-params '{"epochs": 1000}' --epochs 251
#TrivagoLogisticModelTraining_selu____200_3d96e1a1d3


# crm per_item_in_first_pos
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --loss-function-params '{"clip": 15}' --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy per_item_in_first_pos --policy-estimator-extra-params '{"epochs": 1000}' --epochs 251


# Eval
#####################################

# CRM
PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoModelTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____200_4083caf63e" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_probability --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --policy-estimator-extra-params '{"epochs": 1000}' --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"


# BCE

PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoModelTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____200_0534fe1b23" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_probability --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --policy-estimator-extra-params '{"epochs": 1000}' --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"

# CRM Model

PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoModelTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____200_04a4720481" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --policy-estimator-extra-params '{"epochs": 500}' --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"
















 PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoModelTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____200_f2da7472ca" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_logistic_regression_of_pos_item_idx_and_item_ps --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --policy-estimator-extra-params '{"epochs": 1000}' --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"

PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoModelTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____200_f2da7472ca" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_pos_item_idx --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --policy-estimator-extra-params '{"epochs": 1000}' --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"

 
PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoModelTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____200_f2da7472ca" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_logistic_regression_of_pos_item_idx_and_item_ps --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --policy-estimator-extra-params '{"epochs": 1000}' --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"


# BCE
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist":10}' --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function bce --fill-ps-strategy dummy --policy-estimator-extra-params '{"epochs": 1000}' --epochs 251
#TrivagoLogisticModelTraining_selu____200_c89039ba19

PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoModelTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____200_c8ff8d87b5" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --policy-estimator-extra-params '{"epochs": 1000}' --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"