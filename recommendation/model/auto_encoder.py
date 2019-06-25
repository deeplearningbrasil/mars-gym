from typing import List, Callable, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from recommendation.utils import lecun_normal_init


class UnconstrainedAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, encoder_layers: List[int], decoder_layers: List[int], dropout_prob: float,
                 activation_function: Callable = F.selu, weight_init: Callable = lecun_normal_init,
                 dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout):
        super().__init__()

        self.encoder = nn.ModuleList(
            [nn.Linear(
                input_dim if i == 0 else encoder_layers[i - 1],
                layer_size
            ) for i, layer_size in enumerate(encoder_layers)])

        self.decoder = nn.ModuleList(
            [nn.Linear(
                encoder_layers[-1] if i == 0 else decoder_layers[i - 1],
                layer_size
            ) for i, layer_size in enumerate(decoder_layers)])
        self.decoder.append(nn.Linear(decoder_layers[-1], input_dim))

        if dropout_prob:
            self.dropout: nn.Module = dropout_module(dropout_prob)

        self.activation_function = activation_function
        self.weight_init = weight_init
        self.dropout_module = dropout_module

        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = self.activation_function(layer(x))

        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder[:-1]:
            x = self.activation_function(layer(x))
        x = self.decoder[-1](x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.layout == torch.sparse_coo:
            x = x.to_dense()
        return self.decode(self.encode(x))
