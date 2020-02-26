from typing import Tuple, Callable, Union, Type, List
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from recommendation.utils import lecun_normal_init
import numpy as np

class LogisticRegression(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int, vocab_size: int = 88,
                num_filters: int = 32, filter_sizes: List[int] = [1, 3, 5], 
                weight_init: Callable = lecun_normal_init):

        super(LogisticRegression, self).__init__()
        self.weight_init = weight_init
        self.apply(self.init_weights)

        self.user_embeddings        = nn.Embedding(n_users, n_factors)
        self.item_embeddings        = nn.Embedding(n_items, n_factors)
        self.action_type_embeddings = nn.Embedding(10, n_factors)
        self.word_embeddings        = nn.Embedding(vocab_size, n_factors)

        context_embs       = 12
        continuos_features = 3 + 150

        self.convs1  = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (K, n_factors * context_embs)) for K in filter_sizes])
        
        #np.sum([K * num_filters for K in filter_sizes])
        num_dense  = len(filter_sizes) * num_filters  + n_factors * 2 + continuos_features
        self.dense = nn.Sequential(
            nn.Linear(num_dense, int(num_dense/2)),
            nn.ReLU(),
            nn.Linear(int(num_dense/2), n_factors)
        )
        self.output = nn.Linear(n_factors, 1)

        # init
        weight_init(self.action_type_embeddings.weight)
        weight_init(self.user_embeddings.weight)
        weight_init(self.item_embeddings.weight)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def none_tensor(self, x):
        return type(x) == type(None)

    def conv_block(self, x):
		# conv_out.size() = (batch_size, out_channels, dim, 1)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        return x

    def forward(self, user_ids, item_ids, 
                price,  platform_idx, device_idx,
                list_action_type_idx, list_clickout_item_idx,
                list_interaction_item_image_idx, list_interaction_item_info_idx,
                list_interaction_item_rating_idx, list_interaction_item_deals_idx,
                list_search_for_item_idx,
                list_search_for_poi, list_change_of_sort_order, 
                list_search_for_destination, list_filter_selection, 
                list_current_filters, 
                list_metadata):

        # Geral embs
        user_emb           = self.user_embeddings(user_ids)
        item_emb           = self.item_embeddings(item_ids)

        # Categorical embs
        actions_type_emb   = self.action_type_embeddings(list_action_type_idx)

        # Item embs

        # list_reference_clickout_item_idx
        # list_reference_interaction_item_image_idx
        # list_reference_interaction_item_info_idx
        # list_reference_interaction_item_rating_idx
        # list_reference_interaction_item_deals_idx
        # list_reference_search_for_item_idx
                
        clickout_item_emb            = self.item_embeddings(list_clickout_item_idx)
        interaction_item_image_emb   = self.item_embeddings(list_interaction_item_image_idx)
        interaction_item_info_emb    = self.item_embeddings(list_interaction_item_info_idx)
        interaction_item_rating_emb  = self.item_embeddings(list_interaction_item_rating_idx)
        interaction_item_deals_emb   = self.item_embeddings(list_interaction_item_deals_idx)
        search_for_item_emb          = self.item_embeddings(list_search_for_item_idx)

        # NLP embs

        # list_reference_search_for_poi
        # list_reference_change_of_sort_order
        # list_reference_search_for_destination
        # list_reference_filter_selection
        # list_current_filters

        search_for_poi_emb           = self.word_embeddings(list_search_for_poi)
        change_of_sort_order_emb     = self.word_embeddings(list_change_of_sort_order)
        search_for_destination_emb   = self.word_embeddings(list_search_for_destination)
        filter_selection_emb         = self.word_embeddings(list_filter_selection)
        current_filters_emb          = self.word_embeddings(list_current_filters)

        context_session_emb   = torch.cat((actions_type_emb, 
                                    clickout_item_emb, 
                                    interaction_item_image_emb,
                                    interaction_item_info_emb, 
                                    interaction_item_rating_emb, 
                                    interaction_item_deals_emb,
                                    search_for_item_emb, 
                                    search_for_poi_emb, change_of_sort_order_emb,
                                    search_for_destination_emb,
                                    filter_selection_emb, current_filters_emb), dim=2)
        context_session_emb = self.conv_block(context_session_emb)
        #print(context_emb.shape, platform_idx.shape, platform_idx.float().unsqueeze(0).shape, platform_idx.float().unsqueeze(1).shape)
        
        context_emb = torch.cat((context_session_emb, 
                                list_metadata.float(),
                                price.float().unsqueeze(1),
                                platform_idx.float().unsqueeze(1), 
                                device_idx.float().unsqueeze(1)), dim=1)

        x = torch.cat((user_emb, item_emb, context_emb), dim=1)
        x = self.dense(x)

        return torch.sigmoid(self.output(x))

class SimpleLogisticRegression(LogisticRegression):
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
