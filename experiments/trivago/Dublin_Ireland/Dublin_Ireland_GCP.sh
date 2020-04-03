# Model
gcloud ai-platform jobs submit training model_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy model

# Random
gcloud ai-platform jobs submit training random_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy random


# Epsilon Greedy
gcloud ai-platform jobs submit training epsilon_greedy_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.05}' 

gcloud ai-platform jobs submit training epsilon_greedy_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.1}'  

gcloud ai-platform jobs submit training epsilon_greedy_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy epsilon_greedy --bandit-policy-params '{"epsilon": 0.2}'  

# lin_ucb
gcloud ai-platform jobs submit training lin_ucb_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1e-5}'   

gcloud ai-platform jobs submit training lin_ucb_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1e-2}'   

gcloud ai-platform jobs submit training lin_ucb_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1e-1}'   

gcloud ai-platform jobs submit training lin_ucb_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1}'   

# custom_lin_ucb
gcloud ai-platform jobs submit training custom_lin_ucb_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1e-5}'  

gcloud ai-platform jobs submit training custom_lin_ucb_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1e-2}'  

gcloud ai-platform jobs submit training custom_lin_ucb_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1e-1}'  

gcloud ai-platform jobs submit training custom_lin_ucb_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy custom_lin_ucb --bandit-policy-params '{"alpha": 1}'  

# Lin TS
gcloud ai-platform jobs submit training lin_ts_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy lin_ts --bandit-policy-params '{"v_sq": 0.1}' 

gcloud ai-platform jobs submit training lin_ts_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy lin_ts --bandit-policy-params '{"v_sq": 0.5}' 

gcloud ai-platform jobs submit training lin_ts_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy lin_ts --bandit-policy-params '{"v_sq": 1}' 


## softmax_explorer
gcloud ai-platform jobs submit training softmax_explorer_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 0.1}' 

gcloud ai-platform jobs submit training softmax_explorer_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 0.5}' 

gcloud ai-platform jobs submit training softmax_explorer_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 1}'  

gcloud ai-platform jobs submit training softmax_explorer_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy softmax_explorer --bandit-policy-params '{"logit_multiplier": 5.0}'  

## Percentile_adaptive
gcloud ai-platform jobs submit training percentile_adaptive_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy percentile_adaptive --bandit-policy-params '{"exploration_threshold": 0.2}'  

gcloud ai-platform jobs submit training percentile_adaptive_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy percentile_adaptive --bandit-policy-params '{"exploration_threshold": 0.5}'  

gcloud ai-platform jobs submit training percentile_adaptive_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy percentile_adaptive --bandit-policy-params '{"exploration_threshold": 0.7}'  


## Adaptative
## 
gcloud ai-platform jobs submit training adaptive_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.7, "decay_rate": 0.000199566512577}'  

gcloud ai-platform jobs submit training adaptive_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.5, "decay_rate": 0.000199566512577}'  

gcloud ai-platform jobs submit training adaptive_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.3, "decay_rate": 0.000199566512577}'  

#https://www.wolframalpha.com/input/?i=0.1%3D0.7%281-r%29%5E20000
gcloud ai-platform jobs submit training adaptive_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.7, "decay_rate": 0.0000972907743983833}'  

#https://www.wolframalpha.com/input/?i=0.1%3D0.5%281-r%29%5E20000
gcloud ai-platform jobs submit training adaptive_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.5, "decay_rate": 0.0000804686578455631}'  

#https://www.wolframalpha.com/input/?i=0.1%3D0.3%281-r%29%5E20000
gcloud ai-platform jobs submit training adaptive_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy adaptive --bandit-policy-params '{"exploration_threshold": 0.3, "decay_rate": 0.0000549291057748284}'  


## Explore the Exploit
# #https://www.wolframalpha.com/input/?i=0.1%3D0.8%281-r%29%5E2000
#
#
gcloud ai-platform jobs submit training explore_then_exploit_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy explore_then_exploit --bandit-policy-params '{"explore_rounds": 1000, "decay_rate": 0.0001872157}'  

gcloud ai-platform jobs submit training explore_then_exploit_$(date +%Y%m%d_%H%M%S%sss) \
  --region us-central1	 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago-3.5  \
  --scale-tier BASIC_GPU \
  -- \
  --module recommendation.task.model.trivago.trivago_logistic_model TrivagoLogisticModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Dublin, Ireland", "window_hist": 10}' --n-factors 50 --learning-rate=0.001 --optimizer adam --metrics '["loss"]' --epochs 250 --obs-batch-size 1000 --val-split-type random --full-refit --early-stopping-patience 5 --batch-size 200 --num-episodes 15 --output-model-dir "gs://deepfood-results-dublin_ireland" --bandit-policy explore_then_exploit --bandit-policy-params '{"explore_rounds": 1000, "decay_rate": 0.000115122627531392}'  