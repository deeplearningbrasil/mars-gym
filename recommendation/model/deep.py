from typing import List, Callable, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from recommendation.utils import lecun_normal_init
#import ipdb; ipdb.set_trace()


# PYTHONPATH="." luigi --module recommendation.task.model.deep DeepTraining --project criteo --local-scheduler --input-d-dim 13 
# --input-c-dim 26 --generator-workers 0 --loss-function bce_loss --n-factors 10 --epochs 30
class DeepCTR(nn.Module):
    def __init__(self, input_d_dim: int, input_c_dim: int,  
                 dropout_prob: float,
                 n_factors: int,
                 size_embs: List[int],
                 dense_layers: List[int],
                 activation_function: Callable = F.selu, weight_init: Callable = lecun_normal_init,
                 dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout):
        super().__init__()

        #self.activation_function = activation_function
        self.weight_init = weight_init
        self.size_embs   = size_embs
        self.n_factors   = n_factors
        self.input_c_dim = input_c_dim

        # Def Dense Layer 
        def dense_layer(in_f, out_f, *args, **kwargs):
            return nn.Sequential(
                nn.Linear(in_f, out_f, *args, **kwargs),
                nn.ReLU()
            )   

        # Def Embedding Layer
        def embs_layer(size, n_factors):
            return nn.Embedding(size, n_factors)

        # Dense Layers
        size_layers = [input_d_dim + (n_factors * input_c_dim), *dense_layers]
        blocks      = [dense_layer(in_f, out_f) for in_f, out_f in zip(size_layers, size_layers[1:])]
        self.dense_block = nn.Sequential(*blocks)

        # Logistic Loss
        self.logistic = nn.Sequential(
            nn.Linear(size_layers[-1], 1),
            nn.Sigmoid()
        )

        self.embs = nn.ModuleList(embs_layer(int(self.size_embs[i] + 1), self.n_factors) for i in range(self.input_c_dim))

        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def forward(self, xd: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:

        # Dense Features
        x = [xd.float()]

        # Categorical Fatures
        for i, emb in enumerate(self.embs):
            x.append(emb(xc[:, i]))
        
        # Join
        x = torch.cat(x, 1)

        x = self.dense_block(x)
        x = self.logistic(x)
        return x

