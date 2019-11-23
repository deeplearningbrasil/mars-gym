from typing import Tuple, Callable, Union, Type, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from recommendation.model.embedding import UserAndItemEmbedding
from recommendation.model.attention import Attention
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

    def predict(self, item_representation, user_representation, context_representation):
        return torch.sigmoid(self.linear(context_representation))


class ContextualBandit(nn.Module):

    def __init__(self, n_users: int, n_items: int, n_factors: int, weight_init: Callable = lecun_normal_init, 
                use_buys_visits: bool = False, user_embeddings: bool = False, item_embeddings: bool = False, use_numerical_content: bool = False,
                numerical_content_dim: int = None, use_categorical_content: bool = False, context_embeddings: bool = False,
                use_textual_content: bool = False, use_normalize: bool = False, content_layers=[1], binary: bool = False,
                activation_function: Callable = F.selu, predictor: str = "logistic_regression"):
        super(ContextualBandit, self).__init__()


        self.binary = binary
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.context_embeddings = context_embeddings
        self.use_numerical_content = use_numerical_content
        self.use_textual_content = use_textual_content
        self.use_categorical_content = use_categorical_content
        self.use_normalize = use_normalize
        self.use_buys_visits = use_buys_visits
        self.activation_function = activation_function
        self.predictor = predictor


        if self.user_embeddings:
            self.user_embeddings = nn.Embedding(n_users, n_factors)
            weight_init(self.user_embeddings.weight)

        if self.item_embeddings:
            self.item_embeddings = nn.Embedding(n_items, n_factors)
            weight_init(self.item_embeddings.weight)

            # if self.use_textual_content:
            #     self.word_embeddings = nn.Embedding(vocab_size, word_embeddings_size)

            #     self.convs1  = nn.ModuleList(
            #     [nn.Conv2d(1, num_filters, (K, word_embeddings_size)) for K in filter_sizes])
            #     num_dense += np.sum([K * num_filters for K in filter_sizes]) 
            weight_init(self.word_embeddings.weight)

        num_dense = 0
        if self.context_embeddings:
            if self.use_buys_visits:
                num_dense += 2

            if self.use_numerical_content:
                num_dense += numerical_content_dim
            

        # self.content_network = nn.ModuleList(
        #     [nn.Linear(
        #         num_dense if i == 0 else content_layers[i - 1],
        #         layer_size
        #     ) for i, layer_size in enumerate(content_layers)] + 
        #     [nn.Linear(content_layers[-1], n_factors)])

        if predictor == "logistic_regression":
            self.predictor = LogisticRegression(num_dense, weight_init)
        elif predictor == "factorization_machine":
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.weight_init = weight_init

        self.apply(self.init_weights)
        
        

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def conv_block(self, x):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        return x

    def normalize(self, x):
        if type(x) != type(None):
            x = F.normalize(x, p=2, dim=1)    
        return x 

    def compute_item_embeddings(self, item_ids, name, description, category):

        x : torch.Tensor = None

        if self.use_textual_content:
            emb_name, emb_description, emb_category = self.word_embeddings(name), \
                                                        self.word_embeddings(description), \
                                                            self.word_embeddings(category)

            cnn_category    = self.conv_block(emb_category)
            cnn_description = self.conv_block(emb_description)
            cnn_name        = self.conv_block(emb_name)

            x = torch.cat((cnn_category, cnn_description, cnn_name), dim=1)

        if self.item_embeddings:
            item_embs = self.item_embeddings(item_ids)
            x = item_embs if x == None else torch.cat((x, item_embs), dim=1)

        if self.use_normalize:
            x = self.normalize(x)

        return x

    def compute_context_embeddings(self, info, visits, buys):
        x : torch.Tensor = None

        if self.use_categorical_content:
            x = info if x == None else torch.cat((x, info), dim=1)

        if self.use_buys_visits:
            x = torch.cat((visits.unsqueeze(1), buys.unsqueeze(1)), dim=1) if x == None else torch.cat((x, visits.unsqueeze(1), buys.unsqueeze(1)), dim=1)
        
        if self.use_normalize:
            x = self.normalize(x)

        return x


    def compute_user_embeddings(self, user_ids):
        user_emb = self.user_embeddings(user_ids)

        if self.use_normalize:
            out = self.normalize(user_emb)
        
        return out

    def forward(self, user_ids: torch.Tensor, item_content: torch.Tensor, positive_visits: torch.Tensor = None, 
                positive_buys: torch.Tensor = None) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:

        item_ids, name, description, category, info, visits, buys = item_content

        context_representation = self.compute_context_embeddings(info, visits, buys)
        item_representation = self.compute_item_embeddings(item_ids, name, description, category)
        user_representation = self.user_embeddings(user_ids) if self.user_embeddings else None

        prob = self.predictor.predict(item_representation, user_representation, context_representation)
        
        return prob, positive_visits, positive_buys
