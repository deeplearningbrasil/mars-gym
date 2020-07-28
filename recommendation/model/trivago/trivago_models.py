from typing import Tuple, Callable, Union, Type, List
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from recommendation.utils import lecun_normal_init
import numpy as np
from recommendation.model.transformer import *
import copy

class TestModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int, 
                window_hist_size: int, metadata_size: int,
                vocab_size: int = 88,
                num_filters: int = 32, filter_sizes: List[int] = [1, 3, 5], 
                dropout_prob: int = 0.0, dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout,                 
                weight_init: Callable = lecun_normal_init):

        super(TestModel, self).__init__()
        self.weight_init = weight_init
        self.apply(self.init_weights)

        self.user_embeddings        = nn.Embedding(n_users, n_factors)
        self.item_embeddings        = nn.Embedding(n_items, n_factors)
        self.item_embeddings2       = nn.Embedding(n_items, n_factors)

        # Dropout
        self.dropout: nn.Module = dropout_module(dropout_prob)

        num_dense  = 2 * n_factors 
        # output
        self.output = nn.Linear(num_dense, 1)

        self.dense = nn.Sequential(
            nn.Linear(num_dense, int(num_dense/2)),
            nn.ReLU(),
            nn.Linear(int(num_dense/2), 1)
        )
        # init
        weight_init(self.user_embeddings.weight)
        weight_init(self.item_embeddings.weight)
        weight_init(self.item_embeddings2.weight)


    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)
    
    def normalize(self, x):
        x = F.normalize(x, p=2, dim=1)   
        return x   

    def forward(self, user_ids, item_ids, session_idx, action_type_item_idx):
        # Geral embs
        user_emb           = self.user_embeddings(user_ids)
        item_emb           = self.normalize(self.item_embeddings(item_ids))
        item_pred_emb      = self.normalize(self.item_embeddings2(action_type_item_idx))

        x   = torch.cat((item_emb, item_pred_emb), dim=1)

        output = self.dense(x)
        output = torch.sigmoid(output)

        return output

class SimpleLinearModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int, 
                window_hist_size: int, metadata_size: int,
                vocab_size: int = 88,
                num_filters: int = 32, filter_sizes: List[int] = [1, 3, 5], 
                dropout_prob: int = 0.0, dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout,                 
                weight_init: Callable = lecun_normal_init):

        super(SimpleLinearModel, self).__init__()

        self.user_embeddings        = nn.Embedding(n_users, n_factors)
        self.item_embeddings        = nn.Embedding(n_items, n_factors)
        self.action_type_embeddings = nn.Embedding(11, n_factors)
        #self.platform_embeddings    = nn.Embedding(36, n_factors)
        
        self.word_embeddings        = nn.Embedding(vocab_size, n_factors)
        #self.pe                     = PositionalEncoder(n_factors)
        
        # TODO
        context_embs   = 4
        filter_size    = 0
        continuos_size = 4
        # Dropout
        self.dropout: nn.Module = dropout_module(dropout_prob)

        # output
        #raise(Exception(2 * n_factors, context_embs * n_factors * window_hist_size, 2,  metadata_size))
        num_dense  = 1 * n_factors + context_embs * n_factors * window_hist_size + 2 + metadata_size + filter_size
        num_dense  = continuos_size + 1 * n_factors + metadata_size + filter_size + window_hist_size * 6 + context_embs * n_factors * window_hist_size#+ n_factors * n_factors
        
        self.dense = nn.Sequential(
            nn.Linear(num_dense, int(num_dense/2)),
            nn.SELU(),
            nn.Linear(int(num_dense/2), int(num_dense/4)),
            nn.SELU(),
            nn.Linear(int(num_dense/4), 1)
        )

        # init
        weight_init(self.user_embeddings.weight)
        weight_init(self.item_embeddings.weight)
        weight_init(self.action_type_embeddings.weight)
        weight_init(self.word_embeddings.weight)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def flatten(self, input):
        return input.view(input.size(0), -1)        

    def normalize(self, x):
        x = F.normalize(x, p=2, dim=1)   
        return x   

    def item_dot_history(self, itemA, itemB):
        dot  = torch.matmul(self.normalize(itemA.unsqueeze(1)), self.normalize(itemB.permute(0, 2, 1)))
        return self.flatten(dot)
        
    def forward(self, user_ids, item_ids, pos_item_idx,
                    price,  platform_idx, device_idx, 
                    sum_action_item_before, is_first_in_impression,
                    list_action_type_idx, list_clickout_item_idx,
                    list_interaction_item_image_idx, list_interaction_item_info_idx,
                    list_interaction_item_rating_idx, list_interaction_item_deals_idx,
                    list_search_for_item_idx,
                    list_search_for_poi, list_change_of_sort_order, 
                    list_search_for_destination, list_filter_selection, 
                    list_current_filters, 
                    list_metadata):


        # Geral embs
        user_emb                     = self.user_embeddings(user_ids)
        item_emb                     = self.item_embeddings(item_ids)

        # Categorical embs
        actions_type_emb             = self.action_type_embeddings(list_action_type_idx)

        #platform_emb                 = self.platform_embeddings(platform_idx)

        # Item embs
        clickout_item_emb            = self.item_embeddings(list_clickout_item_idx)
        interaction_item_image_emb   = self.item_embeddings(list_interaction_item_image_idx)
        interaction_item_info_emb    = self.item_embeddings(list_interaction_item_info_idx)
        interaction_item_rating_emb  = self.item_embeddings(list_interaction_item_rating_idx) #less
        interaction_item_deals_emb   = self.item_embeddings(list_interaction_item_deals_idx) #less
        search_for_item_emb          = self.item_embeddings(list_search_for_item_idx)

        # NLP embs
        search_for_poi_emb           = self.word_embeddings(list_search_for_poi)
        search_for_destination_emb   = self.word_embeddings(list_search_for_destination)
        change_of_sort_order_emb     = self.word_embeddings(list_change_of_sort_order)
        filter_selection_emb         = self.word_embeddings(list_filter_selection)

        context_session_emb          = self.flatten(
                                            torch.cat((search_for_poi_emb, search_for_destination_emb,
                                                        change_of_sort_order_emb, filter_selection_emb), dim=2))

        # Dot Item X History
        item_dot_clickout_item_emb           = self.item_dot_history(item_emb, clickout_item_emb)
        item_dot_interaction_item_image_emb  = self.item_dot_history(item_emb, interaction_item_image_emb)
        item_dot_interaction_item_info_emb   = self.item_dot_history(item_emb, interaction_item_info_emb)
        item_dot_interaction_item_info_emb   = self.item_dot_history(item_emb, interaction_item_info_emb)
        item_dot_interaction_item_rating_emb = self.item_dot_history(item_emb, interaction_item_rating_emb)
        item_dot_interaction_item_deals_emb  = self.item_dot_history(item_emb, interaction_item_deals_emb)
        item_dot_search_for_item_emb         = self.item_dot_history(item_emb, search_for_item_emb)

        #item_dot_user_emb         = self.item_dot_history(item_emb, user_emb)
        #filter_dot_metadata = (self.filter_dense(list_current_filters.float()) * self.metadata_dense(list_metadata.float())).sum(1)

        #raise(Exception(item_emb.shape, item_clickout_item_emb.shape,item_clickout_item_emb.unsqueeze(1).shape, self.flatten(item_clickout_item_emb).shape))
        x   = torch.cat((item_emb, 
                        item_dot_clickout_item_emb,
                        item_dot_interaction_item_image_emb,
                        item_dot_interaction_item_info_emb,
                        item_dot_interaction_item_rating_emb,
                        item_dot_interaction_item_deals_emb,
                        item_dot_search_for_item_emb,
                        is_first_in_impression.float().unsqueeze(1),
                        pos_item_idx.float().unsqueeze(1),
                        sum_action_item_before.float().unsqueeze(1),
                        price.float().unsqueeze(1),
                        list_metadata.float(),
                        context_session_emb), dim=1)
                
        x   = self.dense(x)
        out = torch.sigmoid(x)
        return out

class SimpleCNNModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int, 
                window_hist_size: int, metadata_size: int,
                vocab_size: int = 88,
                num_filters: int = 32, filter_sizes: List[int] = [1, 3, 5], 
                dropout_prob: int = 0.0, dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout,                 
                weight_init: Callable = lecun_normal_init):

        super(SimpleCNNModel, self).__init__()
        self.weight_init = weight_init
        self.apply(self.init_weights)

        self.user_embeddings        = nn.Embedding(n_users, n_factors)
        self.item_embeddings        = nn.Embedding(n_items, n_factors)
        self.action_type_embeddings = nn.Embedding(11, n_factors)
        self.word_embeddings        = nn.Embedding(vocab_size, n_factors)
        self.pe                     = PositionalEncoder(n_factors)

        # TODO
        context_embs  = 11 # Session Context
        filter_size   = 45

        # Conv
        self.convs1  = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (K, n_factors * context_embs)) for K in filter_sizes])
        
        # Dropout
        self.dropout: nn.Module = dropout_module(dropout_prob)

        # Dense
        num_dense  = len(filter_sizes) * num_filters  + n_factors * 2 + 4 + metadata_size + filter_size
        self.dense = nn.Sequential(
            nn.Linear(num_dense, int(num_dense/2)),
            nn.ReLU(),
            nn.Linear(int(num_dense/2), n_factors)
        )

        # output
        self.output = nn.Linear(n_factors, 1)

        # init
        weight_init(self.user_embeddings.weight)
        weight_init(self.item_embeddings.weight)
        weight_init(self.action_type_embeddings.weight)
        weight_init(self.word_embeddings.weight)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def conv_block(self, x):
		# conv_out.size() = (batch_size, out_channels, dim, 1)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        return x

    def forward(self, user_ids, item_ids, pos_item_idx,
                price,  platform_idx, device_idx, 
                sum_action_item_before, is_first_in_impression,
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
        actions_type_emb   = self.pe(self.action_type_embeddings(list_action_type_idx))

        # Item embs
        clickout_item_emb            = self.pe(self.item_embeddings(list_clickout_item_idx))
        interaction_item_image_emb   = self.pe(self.item_embeddings(list_interaction_item_image_idx))
        interaction_item_info_emb    = self.pe(self.item_embeddings(list_interaction_item_info_idx))
        interaction_item_rating_emb  = self.pe(self.item_embeddings(list_interaction_item_rating_idx))
        interaction_item_deals_emb   = self.pe(self.item_embeddings(list_interaction_item_deals_idx))
        search_for_item_emb          = self.pe(self.item_embeddings(list_search_for_item_idx))

        # NLP embs
        search_for_poi_emb           = self.pe(self.word_embeddings(list_search_for_poi))
        change_of_sort_order_emb     = self.pe(self.word_embeddings(list_change_of_sort_order))
        search_for_destination_emb   = self.pe(self.word_embeddings(list_search_for_destination))
        filter_selection_emb         = self.pe(self.word_embeddings(list_filter_selection))

        context_session_emb   = torch.cat((actions_type_emb, 
                                            clickout_item_emb, 
                                            interaction_item_image_emb,
                                            interaction_item_info_emb, 
                                            interaction_item_rating_emb, 
                                            interaction_item_deals_emb,
                                            search_for_item_emb, 
                                            search_for_poi_emb, 
                                            change_of_sort_order_emb,
                                            search_for_destination_emb,
                                            filter_selection_emb), dim=2)
        context_session_emb = self.conv_block(context_session_emb)

        
        context_emb = torch.cat((user_emb,
                                 item_emb,
                                 is_first_in_impression.float().unsqueeze(1),
                                 pos_item_idx.float().unsqueeze(1),
                                 sum_action_item_before.float().unsqueeze(1),
                                 list_metadata.float(),
                                 list_current_filters.float(),
                                 price.float().unsqueeze(1),
                                 context_session_emb), dim=1)

        x   = self.dropout(context_emb)
        x   = self.dense(x)
        out = torch.sigmoid(self.output(x))
        return out

class SimpleRNNModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int, 
                window_hist_size: int, metadata_size: int,
                vocab_size: int = 88,
                num_filters: int = 32, filter_sizes: List[int] = [1, 3, 5], 
                dropout_prob: int = 0.0, dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout,                 
                weight_init: Callable = lecun_normal_init):

        super(SimpleRNNModel, self).__init__()
        self.weight_init = weight_init
        self.apply(self.init_weights)

        self.user_embeddings        = nn.Embedding(n_users, n_factors)
        self.item_embeddings        = nn.Embedding(n_items, n_factors)
        self.action_type_embeddings = nn.Embedding(11, n_factors)
        self.word_embeddings        = nn.Embedding(vocab_size, n_factors)
        recurrence_hidden_size      = 10

        context_embs       = 11 # Session Context
        filter_size        = 45
        continuos_features = 4 + metadata_size + filter_size + (recurrence_hidden_size * window_hist_size * context_embs)# Price + Meta

        self.rnn1 = nn.GRU(n_factors, recurrence_hidden_size, batch_first=True)
        self.rnn2 = nn.GRU(n_factors, recurrence_hidden_size, batch_first=True)
        self.rnn3 = nn.GRU(n_factors, recurrence_hidden_size, batch_first=True)
        self.rnn4 = nn.GRU(n_factors, recurrence_hidden_size, batch_first=True)
        self.rnn5 = nn.GRU(n_factors, recurrence_hidden_size, batch_first=True)
        self.rnn6 = nn.GRU(n_factors, recurrence_hidden_size, batch_first=True)
        self.rnn7 = nn.GRU(n_factors, recurrence_hidden_size, batch_first=True)
        self.rnn8 = nn.GRU(n_factors, recurrence_hidden_size, batch_first=True)
        self.rnn9 = nn.GRU(n_factors, recurrence_hidden_size, batch_first=True)
        self.rnn10 = nn.GRU(n_factors, recurrence_hidden_size, batch_first=True)
        self.rnn11 = nn.GRU(n_factors, recurrence_hidden_size, batch_first=True)
        self.rnn12 = nn.GRU(n_factors, recurrence_hidden_size, batch_first=True)

        # Conv
        self.convs1  = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (K, n_factors * context_embs)) for K in filter_sizes])
        
        # Dropout
        self.dropout: nn.Module = dropout_module(dropout_prob)

        # Dense
        #np.sum([K * num_filters for K in filter_sizes])
        num_dense  = n_factors * 2 + continuos_features
        self.dense = nn.Sequential(
            nn.Linear(num_dense, int(num_dense/2)),
            nn.ReLU(),
            nn.Linear(int(num_dense/2), n_factors)
        )

        # output
        self.output = nn.Linear(n_factors, 1)

        # init
        weight_init(self.user_embeddings.weight)
        weight_init(self.item_embeddings.weight)
        weight_init(self.action_type_embeddings.weight)
        weight_init(self.word_embeddings.weight)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def conv_block(self, x):
		# conv_out.size() = (batch_size, out_channels, dim, 1)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        return x

    def forward(self, user_ids, item_ids, pos_item_idx,
                price,  platform_idx, device_idx, 
                sum_action_item_before, is_first_in_impression,
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
        actions_type_out, _   = self.rnn12(self.action_type_embeddings(list_action_type_idx))

        # Item embs
        clickout_item_out, _            = self.rnn1(self.item_embeddings(list_clickout_item_idx))
        interaction_item_image_out, _   = self.rnn2(self.item_embeddings(list_interaction_item_image_idx))
        interaction_item_info_out, _    = self.rnn3(self.item_embeddings(list_interaction_item_info_idx))
        interaction_item_rating_out, _  = self.rnn4(self.item_embeddings(list_interaction_item_rating_idx))
        interaction_item_deals_out, _   = self.rnn5(self.item_embeddings(list_interaction_item_deals_idx))
        search_for_item_out, _          = self.rnn6(self.item_embeddings(list_search_for_item_idx))

        # NLP embs
        search_for_poi_out, _           = self.rnn7(self.word_embeddings(list_search_for_poi))
        change_of_sort_order_out, _     = self.rnn8(self.word_embeddings(list_change_of_sort_order))
        search_for_destination_out, _   = self.rnn9(self.word_embeddings(list_search_for_destination))
        filter_selection_out, _         = self.rnn10(self.word_embeddings(list_filter_selection))

        context_session_emb   = torch.cat((actions_type_out, 
                                            clickout_item_out, 
                                            interaction_item_image_out,
                                            interaction_item_info_out, 
                                            interaction_item_rating_out, 
                                            interaction_item_deals_out,
                                            search_for_item_out, 
                                            search_for_poi_out, change_of_sort_order_out,
                                            search_for_destination_out,
                                            filter_selection_out), dim=1)
        
        context_emb = torch.cat((user_emb,
                                 item_emb,
                                 sum_action_item_before.float().unsqueeze(1),
                                 is_first_in_impression.float().unsqueeze(1),
                                 pos_item_idx.float().unsqueeze(1),
                                 context_session_emb.view(len(user_ids), -1), 
                                 list_metadata.float(),
                                 list_current_filters.float(), 
                                 price.float().unsqueeze(1)), dim=1)

        #x   = self.dropout(context_emb)
        x   = self.dense(context_emb)
        out = torch.sigmoid(self.output(x))
        return out

class SimpleCNNTransformerModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int, 
                window_hist_size: int, metadata_size: int,
                vocab_size: int = 88,
                num_filters: int = 32, filter_sizes: List[int] = [1, 3, 5], 
                dropout_prob: int = 0.0, dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout,                 
                weight_init: Callable = lecun_normal_init):

        super(SimpleCNNTransformerModel, self).__init__()
        self.weight_init = weight_init
        self.apply(self.init_weights)

        filter_size        = 45
        context_embs       = 11 # Session Context
        continuos_features = 4 + metadata_size + filter_size# Price + Pos + Meta
        
        
        self.transform_heads = 1
        self.transform_n   = 1

        self.user_embeddings        = nn.Embedding(n_users, n_factors)
        self.item_embeddings        = nn.Embedding(n_items, n_factors)
        self.action_type_embeddings = nn.Embedding(11, n_factors)
        self.word_embeddings        = nn.Embedding(vocab_size, n_factors)
        self.pe                     = PositionalEncoder(n_factors * context_embs)
        self.layers                 = self.get_clones(EncoderLayer(n_factors* context_embs, self.transform_heads, dropout=dropout_prob), self.transform_n)
        self.norm                   = Norm(n_factors* context_embs)


        # Conv
        self.convs1  = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (K, n_factors * context_embs)) for K in filter_sizes])
        
        # Dropout
        self.dropout: nn.Module = dropout_module(dropout_prob)


        # Dense
        #np.sum([K * num_filters for K in filter_sizes])
        num_dense  = len(filter_sizes) * num_filters  + n_factors * 2 +  continuos_features 

        self.dense = nn.Sequential(
            nn.BatchNorm1d(num_features=num_dense),
            nn.Linear(num_dense, int(num_dense/2)),
            nn.ReLU(),
            nn.Linear(int(num_dense/2), n_factors)
        )

        # output
        self.output = nn.Linear(n_factors, 1)

        # init
        weight_init(self.user_embeddings.weight)
        weight_init(self.item_embeddings.weight)
        weight_init(self.action_type_embeddings.weight)
        weight_init(self.word_embeddings.weight)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    #We can then build a convenient cloning function that can generate multiple layers:
    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])        

    def conv_block(self, x):
		# conv_out.size() = (batch_size, out_channels, dim, 1)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        return x

    def layer_transform(self, x, mask = None):
        for i in range(self.transform_n):
            x = self.layers[i](x, mask)
        x   = self.norm(x)
        return x

    def forward(self, user_ids, item_ids, pos_item_idx,
                price,  platform_idx, device_idx,
                sum_action_item_before, is_first_in_impression,
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
        clickout_item_emb            = self.item_embeddings(list_clickout_item_idx)
        interaction_item_image_emb   = self.item_embeddings(list_interaction_item_image_idx)
        interaction_item_info_emb    = self.item_embeddings(list_interaction_item_info_idx)
        interaction_item_rating_emb  = self.item_embeddings(list_interaction_item_rating_idx)
        interaction_item_deals_emb   = self.item_embeddings(list_interaction_item_deals_idx)
        search_for_item_emb          = self.item_embeddings(list_search_for_item_idx)

        # NLP embs
        search_for_poi_emb           = self.word_embeddings(list_search_for_poi)
        change_of_sort_order_emb     = self.word_embeddings(list_change_of_sort_order)
        search_for_destination_emb   = self.word_embeddings(list_search_for_destination)
        filter_selection_emb         = self.word_embeddings(list_filter_selection)

        context_session_emb   = torch.cat((actions_type_emb, 
                                            clickout_item_emb, 
                                            interaction_item_image_emb,
                                            interaction_item_info_emb, 
                                            interaction_item_rating_emb, 
                                            interaction_item_deals_emb,
                                            search_for_item_emb, 
                                            search_for_poi_emb, change_of_sort_order_emb,
                                            search_for_destination_emb,
                                            filter_selection_emb), dim=2)

        # Create transform mask
        #mask = (context_session_emb != 0).unsqueeze(-2)
        context_session_emb = self.pe(context_session_emb)
        context_session_emb = self.layer_transform(context_session_emb, None)
        context_session_emb = self.conv_block(context_session_emb)
        

        context_emb = torch.cat((user_emb,
                                 item_emb,
                                 sum_action_item_before.float().unsqueeze(1),
                                 is_first_in_impression.float().unsqueeze(1),
                                 pos_item_idx.float().unsqueeze(1),
                                 list_metadata.float(),
                                 list_current_filters.float(), 
                                 price.float().unsqueeze(1),
                                 context_session_emb), dim=1)

        x   = self.dropout(context_emb)
        x   = self.dense(x)
        out = torch.sigmoid(self.output(x))
        return out