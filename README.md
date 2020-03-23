# Recommendation-System

## Install

```
conda env create -f environment.yml
```

## Tools

### DataViz

```
streamlit run tools/eval_viz/app.py
```

## Baseline

### Random Model

Avalia um modelo de recomendação randômica

```
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateRandomIfoodModel --local-scheduler 
```

### Most Popular

Avalia um modelo de recomendação que usa os restaurantes mais populares para recomendar 

```
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateMostPopularIfoodModel --local-scheduler 
```

### Most Popular Per User

Avalia um modelo de recomendação que usa os restaurantes mais populares do usuário para fazer a recomendação

```
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateMostPopularPerUserIfoodModel --local-scheduler 
```

## Matrix Factorization

### Matrix Factorization 

#### Train

``` 
PYTHONPATH="." luigi --module recommendation.task.model.matrix_factorization MatrixFactorizationTraining \
--project ifood_binary_buys_cf --n-factors 256 --binary --loss-function bce --metrics '["loss", "acc"]' --epochs 200 --local-scheduler
``` 

``` 
PYTHONPATH="." luigi --module recommendation.task.model.matrix_factorization MatrixFactorizationTraining \
--project ifood_binary_buys_cf_with_random_negative --n-factors 256 --binary --loss-function bce --metrics '["loss", "acc"]' --epochs 200 --local-scheduler
``` 

#### Evaluate

```
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --local-scheduler \
--model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationTraining \
--model-task-id=MatrixFactorizationTraining____500_False_40c44eeb5b
```

### Matrix Factorization With Shift Information

#### Train

```
PYTHONPATH="." luigi --module recommendation.task.model.matrix_factorization MatrixFactorizationWithShiftTraining \
--project ifood_binary_buys_with_shift_cf --n-factors 100 --metrics '["loss", "acc"]' \
--loss-function bce --optimizer-params '{"weight_decay": 1e-8}' --epochs 200 --local-scheduler
```

```
PYTHONPATH="." luigi --module recommendation.task.model.matrix_factorization MatrixFactorizationWithShiftTraining \
--project ifood_binary_buys_with_shift_cf_with_random_negative --n-factors 100 --metrics '["loss", "acc"]' \
--loss-function bce --optimizer-params '{"weight_decay": 1e-8}' --epochs 200 --local-scheduler
```

#### Evaluate

```
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel \
--model-module=recommendation.task.model.matrix_factorization --model-cls=MatrixFactorizationWithShiftTraining \
--model-task-id=MatrixFactorizationWithShiftTraining____500____5ce29b2464 --local-scheduler
```

## AutoEncoders

Arquiteturas de Autoencoders

### CDAE - Collaborative Denoising Auto-Encoders

> Yao Wu, Christopher DuBois, Alice X. Zheng, Martin Ester. 
> Collaborative Denoising Auto-Encoders for Top-N Recommender Systems.
> The 9th ACM International Conference on Web Search and Data Mining (WSDM'16), p153--162, 2016.  
> http://alicezheng.org/papers/wsdm16-cdae.pdf

#### Train

```
PYTHONPATH="." luigi --module recommendation.task.model.auto_encoder UnconstrainedAutoEncoderTraining \
--project ifood_user_cdae --binary --optimizer adam --learning-rate 1e-4 --batch-size 500 \
--generator-workers 0 --loss-function focal --loss-function-params '{"gamma": 10.0, "alpha": 1664.0}' \
--loss-wrapper none --data-frames-preparation-extra-params '{"split_per_user": "True"}' \
--gradient-norm-clipping 2.0 --gradient-norm-clipping-type 1 --data-transformation support_based --epochs 200 --local-scheduler
```

#### Evaluate

```
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateAutoEncoderIfoodModel \
--model-module=recommendation.task.model.auto_encoder --model-cls=UnconstrainedAutoEncoderTraining \
--model-task-id=UnconstrainedAutoEncoderTraining_selu____500_4471d69030 --local-scheduler 
```

### CVAE - Collaborative Variational Auto-Encoders

> Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, Tony Jebara,
> Variational autoencoders for collaborative filtering, 2018.
> https://arxiv.org/abs/1802.05814


#### Train

``` 
PYTHONPATH="." luigi --module recommendation.task.model.auto_encoder VariationalAutoEncoderTraining \
--project ifood_user_cdae --generator-workers=0 --local-scheduler --batch-size=500 --optimizer=adam \
--lr-scheduler=step --lr-scheduler-params='{"step_size": 5, "gamma": 0.8}'  --activation-function=selu \
--encoder-layers=[600,200] --decoder-layers=[600] --loss-function=vae_loss --loss-function-params='{"anneal": 0.01}'  \
--data-transformation=support_based --data-frames-preparation-extra-params='{"split_per_user": "True"}' \
--epochs=200 --learning-rate=0.001
``` 

#### Evaluate

```
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateAutoEncoderIfoodModel --model-module=recommendation.task.model.auto_encoder \
--model-cls=VariationalAutoEncoderTraining --model-task-id=VariationalAutoEncoderTraining_selu____500_249fd09ed1 --local-scheduler 
```

## Triplet Models

### Triplet Net

#### Train

```
PYTHONPATH="." luigi --module recommendation.task.model.triplet_net TripletNetTraining \
--project ifood_binary_buys_triplet_with_random_negative  --batch-size=512 --optimizer=adam \
--lr-scheduler=step --lr-scheduler-params='{"step_size": 5, "gamma": 0.8123}' --learning-rate=0.001 \
--epochs=200 --n-factors=100 --loss-function=weighted_triplet --loss-function-params='{"balance_factor": 2500.0}' --local-scheduler
```

#### Evaluate

```
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel \
--model-module=recommendation.task.model.triplet_net --model-cls=TripletNetTraining \
--model-task-id=TripletNetTraining____512____8f8a418af8 --local-scheduler 
```

### Triplet Content Net

#### Train

```
PYTHONPATH="." luigi --module recommendation.task.model.triplet_net TripletNetSimpleContentTraining \
--project ifood_binary_buys_content_triplet_with_random_negative  --batch-size=512 \
--optimizer=adam --lr-scheduler=step --lr-scheduler-params='{"step_size": 5, "gamma": 0.8123}' \
--learning-rate=0.0001 --epochs=200 --n-factors=100 --loss-function=weighted_triplet \
--loss-function-params='{"balance_factor": 0.9}' --content-layers="[64,10]" --local-scheduler
```

#### Evaluate

```
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodModel --model-module=recommendation.task.model.triplet_net --model-cls=TripletNetSimpleContentTraining --model-task-id=TripletNetSimpleContentTraining_selu____512_a04b6da0d7 --local-scheduler --batch-size 1000
```

### Triplet Content Item-Item 

#### Train

```
PYTHONPATH="."  luigi --module recommendation.task.model.triplet_net TripletNetItemSimpleContentTraining \
--project ifood_session_triplet_with_random_negative --n-factors 100 --loss-function relative_triplet \
--negative-proportion 1 --batch-size 500 --save-item-embedding-tsv --use-normalize --num-filters 64 \
--filter-sizes "[1,3,5]" --save-item-embedding-tsv --optimizer "radam"  \
--content-layers "[64]" --dropout-prob 0.4 --epochs 50 --local-scheduler 
```

#### Evaluate

```
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodTripletNetInfoContent \
--model-module=recommendation.task.model.triplet_net --model-cls=TripletNetItemSimpleContentTraining \
--model-task-id=TripletNetItemSimpleContentTraining_selu____100_1fc9e9d515 --batch-size 600 \
--group-last-k-merchants 2 --local-scheduler
```

### Contextual Bandits

#### Train


```
PYTHONPATH="." luigi --module recommendation.task.model.contextual_bandits ContextualBanditsTraining --project ifood_contextual_bandit --local-scheduler --batch-size=512 --optimizer=radam --lr-scheduler=step --lr-scheduler-params='{"step_size": 5, "gamma": 0.801}' --learning-rate=0.001  --loss-function=crm --use-normalize --use-buys-visits  --content-layers=[256,128,64]  --binary --predictor=logistic_regression --context-embeddings --use-numerical-content --user-embeddings --n-factors=100 --epochs 200
```

#### Evaluate

```
PYTHONPATH="." luigi --module recommendation.task.ifood EvaluateIfoodFullContentModel --local-scheduler  --model-module=recommendation.task.model.contextual_bandits --model-cls=ContextualBanditsTraining --model-task-id=ContextualBanditsTraining_selu____512_1741ef11c6
```

## Docker

docker build . --tag gcr.io/deepfood/deep-reco-gym

docker run -it gcr.io/deepfood/deep-reco-gym:trivago TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 100 --bandit-policy epsilon_greedy

gcloud docker -- push gcr.io/deepfood/deep-reco-gym

gcloud compute disks create deep-reco-gym-output-3 \
    --zone=us-central1-a \
    --source-snapshot deep-reco-gym-snap-1

# f1-micro g1-small

gcloud compute instances create-with-container deep-reco-gym-3 \
    --machine-type n1-standard-2 \
    --zone us-west1-b \
    --container-image gcr.io/deepfood/deep-reco-gym \
    --boot-disk-size 50 \
    --container-restart-policy=never \
    --maintenance-policy TERMINATE \
    --tags=allow-8501 \
    --accelerator="type=nvidia-tesla-k80,count=1" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata='install-nvidia-driver=True,proxy-mode=project_editors' \
    --container-arg="--module" \
    --container-arg="recommendation.task.model.trivago.trivago_models" \
    --container-arg="TrivagoModelInteraction" \
    --container-arg="--project" \
    --container-arg="trivago_contextual_bandit" \
    --container-arg="--data-frames-preparation-extra-params" \
    --container-arg="{\"filter_city\": \"Lausanne, Switzerland\"}" \
    --container-arg="--bandit-policy" \
    --container-arg="model"

gcloud compute instances update-container deep-reco-gym-1 \
    --zone us-central1-a \
    --container-image gcr.io/deepfood/deep-reco-gym \
    --container-restart-policy=never \
    --container-arg="--module" \
    --container-arg="recommendation.task.model.trivago.trivago_models" \
    --container-arg="TrivagoModelInteraction" \
    --container-arg="--project" \
    --container-arg="trivago_contextual_bandit" \
    --container-arg="--bandit-policy" \
    --container-arg="random"

# https://cloud.google.com/ai-platform/training/docs/using-gpus
# https://cloud.google.com/sdk/gcloud/reference/ai-platform/jobs/submit/training
# https://cloud.google.com/compute/docs/machine-types?hl=pt-br
export MODEL_DIR=deep_reco_gym_$(date +%Y%m%d_%H%M%S)
gcloud ai-platform jobs submit training $MODEL_DIR \
  --region us-west2 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago  \
  --scale-tier custom \
  --master-machine-type n1-standard-1	 \
  --master-accelerator count=1,type=nvidia-tesla-p4 \
  -- \
  TrivagoModelInteraction \
  --module=recommendation.task.model.trivago.trivago_models \
  --project=trivago_contextual_bandit \
  --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' \
  --n-factors 50 \
  --learning-rate 0.0001 \
  --optimizer radam \
  --epochs 250 \
  --full-refit \
  --obs-batch-size 100 \
  --early-stopping-patience 10 \
  --batch-size 20 \
  --num-episodes 100 \
  --bandit-policy epsilon_greedy \
  --output-model-dir "gs://deepfood-results"


export MODEL_DIR=deep_reco_gym_$(date +%Y%m%d_%H%M%S)
gcloud ai-platform jobs submit training $MODEL_DIR \
  --region us-west2 \
  --master-image-uri gcr.io/deepfood/deep-reco-gym:trivago  \
  --scale-tier custom \
  --master-machine-type n1-highmem-2	 \
  --master-accelerator count=1,type=nvidia-tesla-p4 \
  -- \
  TrivagoModelInteraction \
  --module=recommendation.task.model.trivago.trivago_models \
  --project=trivago_contextual_bandit \
  --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' \
  --n-factors 50 \
  --learning-rate 0.001 \
  --optimizer adam \
  --epochs 250 \
  --obs-batch-size 200 \
  --early-stopping-patience 10 \
  --batch-size 200 \
  --num-episodes 200 \
  --bandit-policy softmax_explorer \
  --bandit-policy-params '{"logit_multiplier": 5.0}' \
  --output-model-dir "gs://deepfood-results"


docker run --gpus '"device=0"' -it gcr.io/deepfood/deep-reco-gym:trivago --module recommendation.task.model.trivago.trivago_models TrivagoModelInteraction --project trivago_contextual_bandit --data-frames-preparation-extra-params '{"filter_city": "Como, Italy"}' --n-factors 50 --learning-rate=0.0001 --optimizer radam --metrics '["loss"]' --epochs 250 --full-refit --obs-batch-size 100 --early-stopping-patience 10 --batch-size 20 --num-episodes 200 --bandit-policy lin_ucb --bandit-policy-params '{"alpha": 1e-5}'  --output-model-dir "gs://deepfood-results" > nohup4.1 &