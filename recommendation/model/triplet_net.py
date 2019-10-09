from typing import Tuple, Callable, Union, Type, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from recommendation.model.embedding import UserAndItemEmbedding
from recommendation.model.attention import Attention
from recommendation.utils import lecun_normal_init


class TripletNet(UserAndItemEmbedding):

    def __init__(self, n_users: int, n_items: int, n_factors: int,
                 weight_init: Callable = lecun_normal_init):
        super().__init__(n_users, n_items, n_factors, weight_init)

    def forward(self, user_ids: torch.Tensor, positive_item_ids: torch.Tensor,
                negative_item_ids: torch.Tensor = None) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        if negative_item_ids is None:
            return torch.cosine_similarity(self.user_embeddings(user_ids), self.item_embeddings(positive_item_ids))
        return self.user_embeddings(user_ids), self.item_embeddings(positive_item_ids), \
               self.item_embeddings(negative_item_ids)


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
            return torch.cosine_similarity(user_emb), positive_item_emb
        
        negative_item_emb = self.compute_item_embeddings(negative_item_content)

        return self.user_embeddings(user_ids), positive_item_emb, \
               negative_item_emb
        
