from typing import List

import luigi
import numpy as np

# - Para múltiplas GPUs:
## ./cuda_luigid --background --pidfile /tmp/luigid.pid --logdir /tmp/luigid_log
## PYTHONPATH="." ./cuda_luigi --module enqueue_pneumonia EnqueuePneumonia --seed 42 --workers 4
# - Para uma única GPU:
## PYTHONPATH="." luigi --module enqueue_pneumonia EnqueuePneumonia --seed 42 --local-scheduler
from recommendation.task.model.auto_encoder import UnconstrainedAutoEncoderTraining, AttentiveVariationalAutoEncoderTraining
from recommendation.task.model.triplet_net import TripletNetContentTraining, TripletNetTraining
from recommendation.task.model.matrix_factorization import MatrixFactorizationTraining

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pycallgraph import Config
from pycallgraph import GlobbingFilter

config = Config()
config.trace_filter = GlobbingFilter(exclude=[
    'pycallgraph.*',
    '*.secret_function',
])

def track_function():
  model = TripletNetTraining(project="ifood_binary_buys_triplet_with_random_negative",
                    batch_size=512, 
                    epochs=1, 
                    n_factors=100, 
                    loss_function="weighted_triplet",
                    loss_function_params={"balance_factor": 2500.0})

  # model = UnconstrainedAutoEncoderTraining(
  #               project="ifood_user_cdae",
  #               binary=True,
  #               epochs=1, 
  #               optimizer="adam",
  #               generator_workers=0,
  #               loss_function="focal",
  #               loss_function_params={"gamma": 10.0, "alpha": 1616.0},
  #               loss_wrapper="none",
  #               data_frames_preparation_extra_params={"split_per_user": True},
  #               gradient_norm_clipping=2.0,
  #               gradient_norm_clipping_type=1,
  #               data_transformation="support_based",
  #           )                    
                    
  model.run()

graphviz = GraphvizOutput(output_file='output/profile_graph.png')
with PyCallGraph(output=graphviz, config=config):
  track_function()
