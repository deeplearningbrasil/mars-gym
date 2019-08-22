from typing import List, Callable, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from recommendation.utils import lecun_normal_init

class WideCTR(nn.Module):
    def __init__(self, input_dim: int,  dropout_prob: float,
                 activation_function: Callable = F.selu, weight_init: Callable = lecun_normal_init,
                 dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout):
        super().__init__()

        #self.activation_function = activation_function
        self.weight_init         = weight_init

        # Logistic Loss
        self.logistic = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def forward(self, xd: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        
        x = torch.cat((xd.float(), xc.float()), 1)
        #x = x.view(-1)
        x = self.logistic(x)

        return x
