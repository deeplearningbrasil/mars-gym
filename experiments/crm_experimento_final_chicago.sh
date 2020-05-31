# !bin/bash
var_filter_city='Chicago, USA'
n_factors=50 
epochs=250 
batch_size=100 
learning_rate=0.0001 
optimizer=adam 
early_stopping_patience=5 
bucket='gs://result-rio-brazil-crm-offpolicy'


### EVAL



models=( TrivagoLogisticModelTraining_selu____model_e058a4a3de TrivagoLogisticModelTraining_selu____model_6ccd6dc04a TrivagoLogisticModelTraining_selu____model_06b7d18580 TrivagoLogisticModelTraining_selu____model_54330c9cea TrivagoLogisticModelTraining_selu____model_e3d6670bf2 TrivagoLogisticModelTraining_selu____model_a28fc6c480 TrivagoLogisticModelTraining_selu____model_effc5e1960 TrivagoLogisticModelTraining_selu____model_edef58e79d TrivagoLogisticModelTraining_selu____model_fc6b208cb1 TrivagoLogisticModelTraining_selu____model_9cc17b7754 TrivagoLogisticModelTraining_selu____model_3f71748a2b TrivagoLogisticModelTraining_selu____model_199da942b6 TrivagoLogisticModelTraining_selu____model_568773dc19 TrivagoLogisticModelTraining_selu____model_a8c165b2c9 TrivagoLogisticModelTraining_selu____model_accec28c65 TrivagoLogisticModelTraining_selu____model_7bdd79deb4 TrivagoLogisticModelTraining_selu____model_1fee7ba160 TrivagoLogisticModelTraining_selu____model_08662cb88a TrivagoLogisticModelTraining_selu____model_83e821c0d2 TrivagoLogisticModelTraining_selu____model_9b761199e2 TrivagoLogisticModelTraining_selu____model_d2e7668a05 TrivagoLogisticModelTraining_selu____model_e71db1a7ed TrivagoLogisticModelTraining_selu____model_d4c876fbe0 TrivagoLogisticModelTraining_selu____model_d7500dab00 )

for m in ${models[@]}; do
PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id $m --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"
done

####

for i in $(seq 1 5) 
do

#BCE 
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}' --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed $i  --test-size 0.2 --loss-function bce --epochs 100
#

# CRM + Model 
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed $i  --test-size 0.2 --loss-function crm --fill-ps-strategy model --epochs 100

# CRM + Model 
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed $i  --test-size 0.2 --loss-function crm --loss-function-params '{"clip":100}' --fill-ps-strategy model --epochs 100

done

for i in $(seq 1 5) 
do

# CRM + Model 
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --loss-function-params '{"clip":1}'  --fill-ps-strategy model --epochs 100

# CRM + Model 
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --loss-function-params '{"clip":5}'  --fill-ps-strategy model --epochs 100

# CRM + Model 
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --loss-function-params '{"clip":10}'  --fill-ps-strategy model --epochs 100

# CRM + Model 
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --loss-function-params '{"clip":20}'  --fill-ps-strategy model --epochs 100

# CRM + Model 
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --loss-function-params '{"clip":40}'  --fill-ps-strategy model --epochs 100

# CRM + Model 
PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --loss-function-params '{"clip":100}'  --fill-ps-strategy model --epochs 100
done

 
# #
# # CRM + Model 
# PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy model --epochs 51

# #TrivagoLogisticModelTraining_selu____model_421844c44a


# PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --loss-function-params '{"clip": 10}' --fill-ps-strategy model --epochs 51
# #TrivagoLogisticModelTraining_selu____model_1c37491a05

# ##### EVAL


# # BCE
# PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____model_0af1725893" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"


# # CRM
# PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____model_421844c44a" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"

# # CRE
# PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____model_1c37491a05" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250 --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"

# # # CRM + Model 
# # PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer   --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy model --epochs 110
# # #

# # # CRM + Model + Clip
# # PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer  --loss-function-params '{"clip": 100}' --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy model --epochs 100

# #
# # # CRM + Model + per_item_in_first_pos
# # PYTHONPATH="." luigi --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelTraining --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Chicago, USA", "window_hist":10}'  --test-split-type time --n-factors $n_factors --learning-rate $learning_rate --optimizer $optimizer --loss-function-params '{"clip": 2000}'  --metrics '["loss"]' --epochs $epochs --early-stopping-patience $early_stopping_patience --batch-size $batch_size  --seed 42  --test-size 0.2 --loss-function crm --fill-ps-strategy per_item_in_first_pos --epochs 111
# # #


# # ## PerProb
# # PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____model_57eb2a07b1" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_prob --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250  --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"

# # PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____model_bcaf3ee80d" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy per_prob --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250  --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"

# # ## Policy

# # # CRM
# # PYTHONPATH="." luigi --module recommendation.task.model.trivago.evaluation EvaluateTrivagoTestSetPredictions --model-module recommendation.task.model.trivago.trivago_logistic_model --model-cls TrivagoLogisticModelTraining --model-task-id "TrivagoLogisticModelTraining_selu____model_57eb2a07b1" --direct-estimator-module recommendation.task.model.trivago.trivago_logistic_model --direct-estimator-cls TrivagoLogisticModelTraining --fill-ps-strategy model --direct-estimator-epochs 100 --direct-estimator-negative-proportion 0.8 --direct-estimator-epochs 250  --eval-cips-cap 15 --fairness-columns "[\"device_idx\"]"

# # #
