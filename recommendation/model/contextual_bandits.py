from typing import Tuple, Callable, Union, Type, List
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from recommendation.utils import lecun_normal_init
import numpy as np


class LogisticRegression(nn.Module):
    def __init__(self, n_factors: int, weight_init: Callable = lecun_normal_init):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_factors, 1)
        self.weight_init = weight_init
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def none_tensor(self, x):
        return type(x) == type(None)

    def predict(self, item_representation, user_representation, context_representation):
        x: torch.Tensor = context_representation
        
        if not self.none_tensor(user_representation):
            x = user_representation if self.none_tensor(x) else torch.cat((x, user_representation), dim=1)
        if not self.none_tensor(item_representation):
            x = item_representation if self.none_tensor(x) else torch.cat((x, item_representation), dim=1)

        return torch.sigmoid(self.linear(x))


class DeepFactorizationMachine(nn.Module):
    def __init__(self, n_factors: int, context_input_dim: int, item_input_dim: int, user_input_dim: int,
                order: int = 1, weight_init: Callable = lecun_normal_init, deep: bool = False, hidden_layers: List[int] = [32]):
        super(DeepFactorizationMachine, self).__init__()
        input_dnn = 0
        
        if context_input_dim > 0:
            self.context = nn.Linear(context_input_dim, n_factors)
            input_dnn += n_factors
        if item_input_dim > 0:
            self.item = nn.Linear(item_input_dim, n_factors)
            input_dnn += n_factors
        if user_input_dim > 0:
            self.user = nn.Linear(user_input_dim, n_factors)
            input_dnn += n_factors

        self.linear = nn.Linear(context_input_dim + item_input_dim + user_input_dim, 1)
        self.weight_init = weight_init
        self.deep = deep
        self.hidden_layers= nn.ModuleList(
            [nn.Linear(
                input_dnn if i == 0 else hidden_layers[i - 1],
                layer_size
            ) for i, layer_size in enumerate(hidden_layers)])
        self.deep_linear_out = nn.Linear(hidden_layers[-1], 1)
        self.order = order
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def none_tensor(self, x):
        return type(x) == type(None)

    def predict(self, item_representation, user_representation, context_representation):
        latent_vectors = []

        if not self.none_tensor(context_representation):
            latent_vectors.append(self.context(context_representation))
        if not self.none_tensor(item_representation):
            latent_vectors.append(self.item(item_representation))
        if not self.none_tensor(user_representation):
            latent_vectors.append(self.user(user_representation))

        x: torch.Tensor = None

        #1st order interactions
        for v in [item_representation, user_representation, context_representation]:
            x = v if self.none_tensor(x) else torch.cat((x, v), dim=1)

        x = self.linear(x)

        

        #higher order interactions
        for k in range(2, self.order + 1):
            for a, b in combinations(latent_vectors, k):
                dot = torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).squeeze(1)
                x = dot if self.none_tensor(x) else torch.cat((x, dot), dim=1)

        

        #deep model
        if self.deep:   
            v = torch.cat(latent_vectors, dim=1)
            for layer in self.hidden_layers:
                v = F.relu(layer(v))
            v = self.deep_linear_out(v)
            x = torch.cat((x, v), dim=1)

        x = torch.sum(x, dim=1)

        return torch.sigmoid(x)


class ContextualBandit(nn.Module):

    def __init__(self, n_users: int, n_items: int, n_factors: int, vocab_size: int, word_embeddings_size: int, num_filters: int = 64, 
                filter_sizes: List[int] = [1, 3, 5], weight_init: Callable = lecun_normal_init, 
                use_buys_visits: bool = False, user_embeddings: bool = False, item_embeddings: bool = False, use_numerical_content: bool = False,
                numerical_content_dim: int = None, context_embeddings: bool = False,
                use_textual_content: bool = False, use_normalize: bool = False, content_layers=[1], binary: bool = False,
                activation_function: Callable = F.selu, predictor: str = "logistic_regression", fm_order: int = 1, fm_deep: bool = False,
                fm_hidden_layers: List[int] = [64, 32]):
        super(ContextualBandit, self).__init__()


        self.binary = binary
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.context_embeddings = context_embeddings
        self.use_numerical_content = use_numerical_content
        self.use_textual_content = use_textual_content
        self.use_normalize = use_normalize
        self.use_buys_visits = use_buys_visits
        self.activation_function = activation_function
        self.predictor = predictor

        user_input_dim = 0
        item_input_dim = 0
        context_input_dim = 0

        if self.user_embeddings:
            self.user_embeddings = nn.Embedding(n_users, n_factors)
            user_input_dim = n_factors
            weight_init(self.user_embeddings.weight)

        if self.item_embeddings:
            self.item_embeddings = nn.Embedding(n_items, n_factors)
            item_input_dim = n_factors
            weight_init(self.item_embeddings.weight)

        if self.use_textual_content:
            self.word_embeddings = nn.Embedding(vocab_size, word_embeddings_size)

            self.convs1  = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (K, word_embeddings_size)) for K in filter_sizes])
            item_input_dim += np.sum([K * num_filters for K in filter_sizes]) 
            weight_init(self.word_embeddings.weight)
            weight_init

        
        if self.context_embeddings:
            if self.use_buys_visits:
                context_input_dim += 2

            if self.use_numerical_content:
                context_input_dim += numerical_content_dim

        if predictor == "logistic_regression":
            self.predictor = LogisticRegression(item_input_dim + context_input_dim + user_input_dim, weight_init)
        elif predictor == "factorization_machine":
            self.predictor = DeepFactorizationMachine(n_factors, context_input_dim, item_input_dim, user_input_dim, fm_order, \
                weight_init, fm_deep, fm_hidden_layers)
        else:
            raise NotImplementedError

        self.weight_init = weight_init

        self.apply(self.init_weights)
        
        
    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def conv_block(self, x):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        return x

    def none_tensor(self, x):
        return type(x) == type(None)

    def normalize(self, x):
        if self.none_tensor(x):
            return x
        return F.normalize(x, p=2, dim=1)    

    def compute_item_embeddings(self, item_ids, name, description, category):

        x : torch.Tensor = None

        if self.use_textual_content:
            emb_name, emb_description, emb_category = self.word_embeddings(name.long()), \
                                                        self.word_embeddings(description.long()), \
                                                            self.word_embeddings(category.long())

            cnn_category    = self.conv_block(emb_category)
            cnn_description = self.conv_block(emb_description)
            cnn_name        = self.conv_block(emb_name)

            x = torch.cat((cnn_category, cnn_description, cnn_name), dim=1)

        if self.item_embeddings:
            item_embs = self.item_embeddings(item_ids.long())
            x = item_embs if self.none_tensor(x) else torch.cat((x, item_embs), dim=1)

        if self.use_normalize:
            x = self.normalize(x)

        return x

    def compute_context_embeddings(self, info, visits, buys):
        x : torch.Tensor = None

        if self.use_numerical_content:
            x = info if self.none_tensor(x) else torch.cat((x, info), dim=1)

        if self.use_buys_visits:
            x = torch.cat((visits.unsqueeze(1), buys.unsqueeze(1)), dim=1) if self.none_tensor(x) else torch.cat((x, visits.unsqueeze(1), buys.unsqueeze(1)), dim=1)
        
        if self.use_normalize:
            x = self.normalize(x)

        return x


    def compute_user_embeddings(self, user_ids):
        user_emb = self.user_embeddings(user_ids.long())

        if self.use_normalize:
            user_emb = self.normalize(user_emb)
        
        return user_emb

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, name: torch.Tensor, description: torch.Tensor,
                category: torch.Tensor, info: torch.Tensor, visits: torch.Tensor, buys: torch.Tensor) -> torch.Tensor:
        context_representation = self.compute_context_embeddings(info, visits, buys)
        item_representation = self.compute_item_embeddings(item_ids, name, description, category)
        user_representation = self.compute_user_embeddings(user_ids) if self.user_embeddings else None

        prob = self.predictor.predict(item_representation, user_representation, context_representation)
        
        return prob


# PYTHONPATH="." luigi --module recommendation.task.model.contextual_bandits DirectEstimatorTraining 
# --project ifood_offpolicy_direct_estimator --batch-size=512 --optimizer=radam --learning-rate=0.0001 
# --loss-function=bce --n-factors=100 --epochs 305 --dropout-prob 0.1 --local-scheduler
#
#
# DirectEstimatorTraining(project='ifood_offpolicy_direct_estimator', batch_size=512, epochs=305, 
# optimizer='radam', learning_rate=0.0001, loss_function='bce', n_factors=100, dropout_prob=0.1)
#
class DirectEstimator(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int, dropout_prob: float,
                    use_normalize: bool = False, weight_init: Callable = lecun_normal_init,
                    dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout):
        super(DirectEstimator, self).__init__()
        self.linear = nn.Linear((n_factors * 3) + 24, 1)
        self.weight_init = weight_init
        self.apply(self.init_weights)

        self.use_normalize   = use_normalize
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)
        self.shift_embeddings = nn.Embedding(10, n_factors)
        
        if dropout_prob:
            self.dropout: nn.Module = dropout_module(dropout_prob)
                
        weight_init(self.user_embeddings.weight)
        weight_init(self.item_embeddings.weight)
        weight_init(self.shift_embeddings.weight)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def none_tensor(self, x):
        return type(x) == type(None)

    def compute_user_embeddings(self, user_ids):
        user_emb = self.user_embeddings(user_ids.long())

        if self.use_normalize:
            user_emb = self.normalize(user_emb)
        
        return user_emb

    def compute_item_embeddings(self, item_ids):
        item_emb = self.item_embeddings(item_ids.long())

        if self.use_normalize:
            item_emb = self.normalize(item_emb)
        
        return item_emb

    def compute_shift_embeddings(self, shift_ids):
        shift_emb = self.shift_embeddings(shift_ids.long())

        if self.use_normalize:
            shift_emb = self.normalize(shift_emb)
        
        return shift_emb

    def compute_context_embeddings(self, info, visits, buys):
        x : torch.Tensor = None

        x = info if self.none_tensor(x) else torch.cat((x, info), dim=1)
        x = torch.cat((visits.unsqueeze(1), buys.unsqueeze(1)), dim=1) if self.none_tensor(x) else torch.cat((x, visits.unsqueeze(1), buys.unsqueeze(1)), dim=1)
        
        return x

    def forward(self, user_ids: torch.Tensor, shift_ids: torch.Tensor, item_ids: torch.Tensor, category: torch.Tensor, 
                    info: torch.Tensor, visits: torch.Tensor, buys: torch.Tensor) -> torch.Tensor:
                    
        context_representation = self.compute_context_embeddings(info, visits, buys)
        shift_representation   = self.compute_shift_embeddings(shift_ids)
        item_representation    = self.compute_item_embeddings(item_ids)
        user_representation    = self.compute_user_embeddings(user_ids) 

        x = torch.cat((context_representation, shift_representation, user_representation, item_representation), dim=1)

        if hasattr(self, "dropout"):
            x = self.dropout(x)

        return torch.sigmoid(self.linear(x))#.reshape(-1)
