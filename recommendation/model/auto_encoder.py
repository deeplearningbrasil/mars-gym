from typing import List, Callable, Type, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from recommendation.utils import lecun_normal_init


class UnconstrainedAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, encoder_layers: List[int], decoder_layers: List[int], binary: bool, dropout_prob: float,
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
        self.decoder.append(nn.Linear(decoder_layers[-1] if decoder_layers else encoder_layers[-1], input_dim))

        if binary:
            self.decoder.append(nn.Sigmoid())

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


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, encoder_layers: List[int], decoder_layers: List[int], binary: bool, dropout_prob: float,
                 activation_function: Callable = F.selu, weight_init: Callable = lecun_normal_init,
                 dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout):
        super().__init__()

        self.decoder_input = encoder_layers[-1] // 2

        self.encoder = nn.ModuleList(
            [nn.Linear(
                input_dim if i == 0 else encoder_layers[i - 1],
                layer_size
            ) for i, layer_size in enumerate(encoder_layers)])

        self.decoder = nn.ModuleList(
            [nn.Linear(
                self.decoder_input if i == 0 else decoder_layers[i - 1],
                layer_size
            ) for i, layer_size in enumerate(decoder_layers)])
        self.decoder.append(nn.Linear(decoder_layers[-1], input_dim))

        if binary:
            self.decoder.append(nn.Sigmoid())

        if dropout_prob:
            self.dropout: nn.Module = dropout_module(dropout_prob)
        
        self.activation_function = activation_function
        self.weight_init = weight_init

        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x)

        if hasattr(self, "dropout"):
            x = self.dropout(x)


        for layer in self.encoder[:-1]:
            x = self.activation_function(layer(x))

        encoder_output = self.encoder[-1](x)

        mu = encoder_output[:, :self.decoder_input]
        logvar = encoder_output[:, self.decoder_input:]
        return mu, logvar

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder[:-1]:
            x = self.activation_function(layer(x))
        x = self.decoder[-1](x)

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.layout == torch.sparse_coo:
            x = x.to_dense()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        return self.decode(z), mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

class HybridVAE(nn.Module):
    def __init__(self, input_dim: int, encoder_layers: List[int], decoder_layers: List[int], embedding_size, embedding_factors: int, binary: bool, dropout_prob: float,
                 activation_function: Callable = F.selu, weight_init: Callable = lecun_normal_init,
                 dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout):

        super().__init__()

        self.encoder_output = encoder_layers[-1] // 2
        self.decoder_input = (encoder_layers[-1] // 2) + embedding_factors

        self.encoder = nn.ModuleList(
            [nn.Linear(
                input_dim if i == 0 else encoder_layers[i - 1],
                layer_size
            ) for i, layer_size in enumerate(encoder_layers)])

        self.decoder = nn.ModuleList(
            [nn.Linear(
                self.decoder_input if i == 0 else decoder_layers[i - 1],
                layer_size
            ) for i, layer_size in enumerate(decoder_layers)])
        self.decoder.append(nn.Linear(decoder_layers[-1], input_dim))

        self.embedding = nn.Embedding(embedding_size, embedding_factors)

        if binary:
            self.decoder.append(nn.Sigmoid())

        if dropout_prob:
            self.dropout: nn.Module = dropout_module(dropout_prob)
        
        self.activation_function = activation_function
        self.weight_init = weight_init

        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def decode(self, x: torch.Tensor, id: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(id)
        x = torch.cat((x, emb.squeeze(1)), dim=1)
        for layer in self.decoder[:-1]:
            x = self.activation_function(layer(x))
        x = self.decoder[-1](x)

        return x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x)

        if hasattr(self, "dropout"):
            x = self.dropout(x)


        for layer in self.encoder[:-1]:
            x = self.activation_function(layer(x))

        encoder_output = self.encoder[-1](x)

        mu = encoder_output[:, :self.encoder_output]
        logvar = encoder_output[:, self.encoder_output:]
        return mu, logvar
    
    def forward(self, x: torch.Tensor, id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.layout == torch.sparse_coo:
            x = x.to_dense()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, id), mu, logvar

class AttentiveVariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, encoder_layers: List[int], attention_layers: List[int], decoder_layers: List[int], binary: bool, dropout_prob: float,
                 activation_function: Callable = F.selu, weight_init: Callable = lecun_normal_init,
                 dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout):
        super().__init__()

        assert encoder_layers[-1] == attention_layers[-1]

        self.encoder_output = encoder_layers[-1]
        self.attention_output = attention_layers[-1]
        self.decoder_input = (self.encoder_output // 2)
        

        self.encoder = nn.ModuleList(
            [nn.Linear(
                input_dim if i == 0 else encoder_layers[i - 1],
                layer_size
            ) for i, layer_size in enumerate(encoder_layers)])

        self.attention = nn.ModuleList(
            [nn.Linear(
                sum(encoder_layers) if i == 0 else attention_layers[i - 1],
                layer_size
            ) for i, layer_size in enumerate(attention_layers)])

        self.decoder = nn.ModuleList(
            [nn.Linear(
                self.decoder_input if i == 0 else decoder_layers[i - 1],
                layer_size
            ) for i, layer_size in enumerate(decoder_layers)])
        self.decoder.append(nn.Linear(decoder_layers[-1], input_dim))

        if binary:
            self.decoder.append(nn.Sigmoid())

        if dropout_prob:
            self.dropout: nn.Module = dropout_module(dropout_prob)
        
        self.activation_function = activation_function
        self.weight_init = weight_init

        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x)

        if hasattr(self, "dropout"):
            x = self.dropout(x)

        att = []
        for layer in self.encoder[:-1]:
            x = self.activation_function(layer(x))
            att.append(x)

        encoder_output = self.encoder[-1](x)
        att.append(encoder_output)

        att = torch.cat(att, dim=-1)

        for layer in self.attention[:-1]:
            att = self.activation_function(layer(att))
        
        att = self.attention[-1](att)

        att_output = 100.0 * F.softmax(att, 1)

        att_mu = att_output[:, :(self.attention_output // 2)]

        att_logvar = att_output[:, (self.attention_output // 2):]

        mu = encoder_output[:, :(self.encoder_output // 2)]
        logvar = encoder_output[:, (self.encoder_output // 2):]
        return mu, logvar, att_mu, att_logvar

    def decode(self, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = a * z
        for layer in self.decoder[:-1]:
            x = self.activation_function(layer(x))
        x = self.decoder[-1](x)

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.layout == torch.sparse_coo:
            x = x.to_dense()
        mu, logvar, att_mu, att_logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        a = self.reparameterize(att_mu, att_logvar)
        return self.decode(a, z), mu, logvar, att_mu, att_logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
