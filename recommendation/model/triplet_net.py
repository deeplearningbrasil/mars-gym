from typing import Tuple, Callable, Union, Type, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from recommendation.model.embedding import UserAndItemEmbedding
from recommendation.model.attention import Attention
from recommendation.utils import lecun_normal_init
import numpy as np

class TripletNet(UserAndItemEmbedding):

    def __init__(self, n_users: int, n_items: int, n_factors: int,
                 weight_init: Callable = lecun_normal_init):
        super().__init__(n_users, n_items, n_factors, weight_init)

    def forward(self, user_ids: torch.Tensor, positive_item_ids: torch.Tensor,
                negative_item_ids: torch.Tensor = None, positive_visits: torch.Tensor = None,
                positive_buys: torch.Tensor = None) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        if negative_item_ids is None:
            return torch.cosine_similarity(self.user_embeddings(user_ids), self.item_embeddings(positive_item_ids))
        
        if positive_visits is None:
            return self.user_embeddings(user_ids.long()), self.item_embeddings(positive_item_ids.long()), \
                self.item_embeddings(negative_item_ids.long())
                
        return self.user_embeddings(user_ids.long()), self.item_embeddings(positive_item_ids.long()), \
                self.item_embeddings(negative_item_ids.long()), positive_visits, positive_buys

class TripletNetItemSimpleContent(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int, word_embeddings_size: int, recurrence_hidden_size: int,  menu_full_text_max_words: int, num_filters: int = 64, filter_sizes: List[int] = [1, 3, 5],
                 dropout_prob: int = 0.1, use_normalize: bool = False, binary: bool = False, dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout, 
                 activation_function: Callable = F.selu, n_factors: int = 128, content_layers: List[int] = [128],  weight_init: Callable = lecun_normal_init):
        super(TripletNetItemSimpleContent, self).__init__()

        self.binary              = binary
        self.use_normalize       = use_normalize
        self.word_embeddings     = nn.Embedding(vocab_size, word_embeddings_size)
        self.activation_function = activation_function

        # RNN Emnedding
        self.lstm = nn.LSTM(word_embeddings_size, recurrence_hidden_size, bidirectional=True, batch_first=True)
        self.attention_menu_full_text = Attention(recurrence_hidden_size * 2, menu_full_text_max_words)

        # Conv1 Embedding
        self.convs1  = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (K, word_embeddings_size)) for K in filter_sizes])

        # Content Information
        self.content_network = nn.ModuleList(
            [nn.Linear(
                input_dim if i == 0 else content_layers[i - 1],
                layer_size
            ) for i, layer_size in enumerate(content_layers)])

        # Dense Layer
        input_dense = (len(filter_sizes) * num_filters) * 3 + content_layers[-1] + recurrence_hidden_size * 2
        self.dense  = nn.Sequential(
            nn.Linear(input_dense, int(input_dense/2)),
            nn.ReLU(),
            nn.Linear(int(input_dense/2), n_factors)
        )

        if dropout_prob:
            self.dropout: nn.Module = dropout_module(dropout_prob)

        self.weight_init = weight_init

        self.apply(self.init_weights)
        weight_init(self.word_embeddings.weight)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def conv_block1(self, x):
		# conv_out.size() = (batch_size, out_channels, dim, 1)
        _x = x.unsqueeze(1)
        _x = [F.relu(conv(_x)).squeeze(3) for conv in self.convs1]
        _x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in _x]
        _x = torch.cat(_x, 1)

        return _x

    def conv_block2(self, x):
        x = self.convs2(x)

        return x

    def normalize(self, x):
        x = F.normalize(x, p=2, dim=1)   

        return x            

    def dense_block(self, x):
        #print("=====> ", x.size())
        x = self.dense(x)

        if self.use_normalize:
            x = self.normalize(x)

        if self.binary:
            x = torch.sigmoid(x)
        
        return x

    def compute_item_embeddings(self, item_content):
        name, description, category, menu_full_text, info = item_content
        #print(name.size(), description.size(), category.size(), menu_full_text.size(), info.size())
        emb_name            = self.word_embeddings(name)
        #emb_description     = self.word_embeddings(description)
        emb_category        = self.word_embeddings(category)
        emb_menu_full_text  = self.word_embeddings(menu_full_text)

        cnn_name            = self.conv_block1(emb_name)
        #cnn_description     = self.conv_block1(emb_description)
        cnn_category        = self.conv_block1(emb_category)
        cnn_menu_full_text  = self.conv_block1(emb_menu_full_text)


        h_menu_full_text, _ = self.lstm(emb_menu_full_text)
        att_menu_full_text  = self.attention_menu_full_text(h_menu_full_text)

        text_out = torch.cat((cnn_name, cnn_category, cnn_menu_full_text, att_menu_full_text), dim=1)
        #x = self.conv_block2(x)

        for layer in self.content_network:
            info = self.activation_function(layer(info.float()))
        
        x = torch.cat((text_out, info), dim=1)

        if hasattr(self, "dropout"):
            x = self.dropout(x)

        emb = self.dense_block(x)

        return emb

    def forward(self, item_content: torch.Tensor, positive_item_content: torch.Tensor,
                negative_item_content: torch.Tensor = None,
                relative_pos: torch.Tensor = None) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        positive_item_emb   = self.compute_item_embeddings(positive_item_content)
        item_emb            = self.compute_item_embeddings(item_content)

        if negative_item_content is None:
            return self.similarity(item_emb, positive_item_emb)
        
        if relative_pos is None:
            negative_item_emb = self.compute_item_embeddings(negative_item_content)
            return item_emb, positive_item_emb, negative_item_emb

        negative_item_emb = self.compute_item_embeddings(negative_item_content)
        return item_emb, positive_item_emb, negative_item_emb, relative_pos

    def similarity(self, emb1, emb2):
        return torch.cosine_similarity(emb1.float(), emb2.float())

class TripletNetSimpleContent(nn.Module):
    def __init__(self, input_dim: int, n_users: int, vocab_size: int, word_embeddings_size: int, num_filters: int = 64, filter_sizes: List[int] = [1, 3, 5],
                 dropout_prob: int = 0.1, binary: bool = False, dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout, content_layers: List[int] = [128],
                 activation_function: Callable = F.selu, n_factors: int = 128, weight_init: Callable = lecun_normal_init):
        super(TripletNetSimpleContent, self).__init__()

        self.binary = binary
        self.word_embeddings = nn.Embedding(vocab_size, word_embeddings_size)
        self.user_embeddings = nn.Embedding(n_users, n_factors)

        self.convs1  = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (K, word_embeddings_size)) for K in filter_sizes])
        
        self.activation_function = activation_function

        # Dense Layer
        num_dense = np.sum([K * num_filters for K in filter_sizes]) + input_dim
        #self.bn1   = nn.BatchNorm1d(num_dense)
        #self.dense = nn.Linear(num_dense, n_factors)
        self.dense = nn.Sequential(
            nn.Linear(num_dense, int(num_dense/2)),
            nn.ReLU(),
            nn.Linear(int(num_dense/2), n_factors)
        )

        if dropout_prob:
            self.dropout: nn.Module = dropout_module(dropout_prob)

        self.weight_init = weight_init

        self.apply(self.init_weights)
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

    def dense_block(self, x):
        x = self.dense(x)

        if self.binary:
            x = torch.sigmoid(x)
        
        return x

    def compute_item_embeddings(self, item_content):
        name, description, category, info = item_content

        emb_name, emb_description, emb_category = self.word_embeddings(name), \
                                                    self.word_embeddings(description), \
                                                        self.word_embeddings(category)

        cnn_category    = self.conv_block(emb_category)
        cnn_description = self.conv_block(emb_description)
        cnn_name        = self.conv_block(emb_name)

        x = torch.cat((cnn_category, cnn_description, cnn_name, info), dim=1)

        if hasattr(self, "dropout"):
            x = self.dropout(x)

        out = self.dense_block(x)

        return out

    def forward(self, user_ids: torch.Tensor, positive_item_content: torch.Tensor,
                negative_item_content: torch.Tensor = None) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        positive_item_emb = self.compute_item_embeddings(positive_item_content)
        user_emb          = self.user_embeddings(user_ids)

        if negative_item_content is None:
            return torch.cosine_similarity(user_emb, positive_item_emb)

        negative_item_emb = self.compute_item_embeddings(negative_item_content)

        return user_emb, positive_item_emb, negative_item_emb


class TripletNetContent(nn.Module):
    def __init__(self, input_dim: int, n_users: int, vocab_size: int, word_embeddings_size: int, max_text_len_description: int, max_text_len_category: int, max_text_len_name: int, recurrence_hidden_size: int = 40, word_embeddings_output: int = 128,
                    dropout_prob: int = 0.1, dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout, content_layers: List[int] = [128], 
                    activation_function: Callable = F.selu, n_factors: int = 128, weight_init: Callable = lecun_normal_init):
        super(TripletNetContent, self).__init__()
   
        self.word_embeddings = nn.Embedding(vocab_size, word_embeddings_size)
        
        self.lstm = nn.LSTM(word_embeddings_size, recurrence_hidden_size, bidirectional=True, batch_first=True)
        
        self.attention_description = Attention(recurrence_hidden_size * 2, max_text_len_description)
        self.attention_category = Attention(recurrence_hidden_size * 2, max_text_len_category)
        self.attention_name = Attention(recurrence_hidden_size * 2, max_text_len_name)
        
        self.linear = nn.Linear(3 * 2 * recurrence_hidden_size, word_embeddings_output)
        self.activation_function = activation_function

        if dropout_prob:
            self.dropout: nn.Module = dropout_module(dropout_prob)

        self.content_network = nn.ModuleList(
            [nn.Linear(
                input_dim if i == 0 else content_layers[i - 1],
                layer_size
            ) for i, layer_size in enumerate(content_layers)])

        self.join_layer = nn.Linear(word_embeddings_output + content_layers[-1], n_factors)

        self.user_embeddings = nn.Embedding(n_users, n_factors)

        self.weight_init = weight_init
        
        self.apply(self.init_weights)
        weight_init(self.word_embeddings.weight)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def compute_item_embeddings(self, item_content):
        name, description, category, info = item_content

        emb_name, emb_description, emb_category = self.word_embeddings(name), self.word_embeddings(description), self.word_embeddings(category)

        h_category, _ = self.lstm(emb_category)
        h_description, _ = self.lstm(emb_description)
        h_name, _ = self.lstm(emb_name)

        att_category = self.attention_category(h_category)
        att_description = self.attention_description(h_description)
        att_name = self.attention_name(h_name)

        text_layer = torch.cat((att_category, att_description, att_name), dim = 1)
        text_out = self.activation_function(self.linear(text_layer))

        for layer in self.content_network:
            info = self.activation_function(layer(info.float()))
        
        x = torch.cat((text_out, info), dim=1)

        if hasattr(self, "dropout"):
            x = self.dropout(x)

        out = self.join_layer(x)

        return out

    def forward(self, user_ids: torch.Tensor, positive_item_content: torch.Tensor,
                negative_item_content: torch.Tensor = None) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        positive_item_emb = self.compute_item_embeddings(positive_item_content)
        user_emb = self.user_embeddings(user_ids)
        
        if negative_item_content is None:
            return torch.cosine_similarity(user_emb, positive_item_emb)
        
        negative_item_emb = self.compute_item_embeddings(negative_item_content)

        return self.user_embeddings(user_ids), positive_item_emb, \
               negative_item_emb
