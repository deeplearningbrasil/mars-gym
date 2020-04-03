# Model
gcloud ai-platform jobs submit training model_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy model

# Random
gcloud ai-platform jobs submit training random_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy random


# Epsilon Greedy
gcloud ai-platform jobs submit training epsilon_greedy_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.05}' 

gcloud ai-platform jobs submit training epsilon_greedy_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.1}'  

gcloud ai-platform jobs submit training epsilon_greedy_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.2}'  

# lin_ucb
gcloud ai-platform jobs submit training lin_ucb_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1e-5}'   

gcloud ai-platform jobs submit training lin_ucb_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1e-2}'   

gcloud ai-platform jobs submit training lin_ucb_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1e-1}'   

gcloud ai-platform jobs submit training lin_ucb_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1}'   

# custom_lin_ucb
gcloud ai-platform jobs submit training custom_lin_ucb_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1e-5}'  

gcloud ai-platform jobs submit training custom_lin_ucb_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1e-2}'  

gcloud ai-platform jobs submit training custom_lin_ucb_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1e-1}'  

gcloud ai-platform jobs submit training custom_lin_ucb_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1}'  

# # Lin TS
# gcloud ai-platform jobs submit training lin_ts_$(date +%Y%m%d_%H%M%S%sss) \
#   --region us-central1	 \
#   --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
#   --scale-tier BASIC \
#   -- \
#   --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy lin_ts --bandit-policy-params '{"v_sq": 0.1}' 

# gcloud ai-platform jobs submit training lin_ts_$(date +%Y%m%d_%H%M%S%sss) \
#   --region us-central1	 \
#   --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
#   --scale-tier BASIC \
#   -- \
#   --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy lin_ts --bandit-policy-params '{"v_sq": 0.5}' 

# gcloud ai-platform jobs submit training lin_ts_$(date +%Y%m%d_%H%M%S%sss) \
#   --region us-central1	 \
#   --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
#   --scale-tier BASIC \
#   -- \
#   --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy lin_ts --bandit-policy-params '{"v_sq": 1}' 


## softmax_explorer
gcloud ai-platform jobs submit training softmax_explorer_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 0.1}' 

gcloud ai-platform jobs submit training softmax_explorer_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 0.5}' 

gcloud ai-platform jobs submit training softmax_explorer_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 1}'  

gcloud ai-platform jobs submit training softmax_explorer_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  

## Percentile_adaptive
gcloud ai-platform jobs submit training percentile_adaptive_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy percentile_adaptive --bandit-policy-params '{"exploration_threshold": 0.2}'  

gcloud ai-platform jobs submit training percentile_adaptive_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy percentile_adaptive --bandit-policy-params '{"exploration_threshold": 0.5}'  

gcloud ai-platform jobs submit training percentile_adaptive_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy percentile_adaptive --bandit-policy-params '{"exploration_threshold": 0.7}'  


## Adaptative
## 
gcloud ai-platform jobs submit training adaptive_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.7, "decay_rate": 0.0000299366311063513}'  

gcloud ai-platform jobs submit training adaptive_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.5, "decay_rate": 0.0000268236054478970}'  

gcloud ai-platform jobs submit training adaptive_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.3, "decay_rate": 0.0000183100371803582}'  

## Explore the Exploit
# #https://www.wolframalpha.com/input/?i=0.1%3D0.8%281-r%29%5E2000
#
#
gcloud ai-platform jobs submit training explore_then_exploit_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Rio de Janeiro, Brazil", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 7 --output-model-dir "gs://deepfood-results-rio_janeiro_brazil" --bandit-policy explore_then_exploit --bandit-policy-params '{"explore_rounds": 1500, "decay_rate": 0.0000416614429241702}'  