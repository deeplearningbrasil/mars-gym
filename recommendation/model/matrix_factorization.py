from typing import Callable, List, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from recommendation.model.embedding import UserAndItemEmbedding
from recommendation.utils import lecun_normal_init


class MatrixFactorization(UserAndItemEmbedding):

    def __init__(self, n_users: int, n_items: int, n_factors: int, binary: bool,
                 weight_init: Callable = lecun_normal_init):
        super().__init__(n_users, n_items, n_factors, weight_init)
        self.binary = binary

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:

        output = (self.user_embeddings(user_ids.long()) * self.item_embeddings(item_ids.long())).sum(1)
        if self.binary:
            return torch.sigmoid(output)
        return output


class BiasedMatrixFactorization(UserAndItemEmbedding):

    def __init__(self, n_users: int, n_items: int, n_factors: int, binary: bool,
                 weight_init: Callable = lecun_normal_init):
        super().__init__(n_users, n_items, n_factors, weight_init)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        self.binary = binary

        self.user_bias.weight.data.zero_()
        self.item_bias.weight.data.zero_()

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_bias = self.user_bias(user_ids).squeeze(1)
        item_bias = self.item_bias(item_ids).squeeze(1)

        dot = (user_embedding * item_embedding).sum(1)

        output = dot + user_bias + item_bias
        if self.binary:
            return torch.sigmoid(output)
        return output


class DeepMatrixFactorization(UserAndItemEmbedding):

    def __init__(self, n_users: int, n_items: int, n_factors: int, binary: bool,
                 dense_layers: List[int], dropout_between_layers_prob: float,
                 bn_between_layers: bool, activation_function: Callable = F.selu,
                 weight_init: Callable = lecun_normal_init,
                 dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout):
        super().__init__(n_users, n_items, n_factors, weight_init)
        self.dense_layers = nn.ModuleList(
            [nn.Linear(
                2 * n_factors if i == 0 else dense_layers[i - 1],
                layer_size
            ) for i, layer_size in enumerate(dense_layers)])
        self.last_dense_layer = nn.Linear(dense_layers[-1], 1)
        if dropout_between_layers_prob:
            self.dropout: nn.Module = dropout_module(dropout_between_layers_prob)

        self.bn_between_layers = bn_between_layers
        self.activation_function = activation_function
        self.binary = binary

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        x = torch.cat((user_embedding, item_embedding), 1)
        for layer in self.dense_layers:
            x = layer(x)
            if self.bn_between_layers:
                x = F.batch_norm(x)
            x = self.activation_function(x)
            if hasattr(self, "dropout"):
                x = self.dropout(x)

        x = self.last_dense_layer(x).flatten()

        if self.binary:
            return torch.sigmoid(x)
        return x


class MatrixFactorizationWithShift(UserAndItemEmbedding):

    def __init__(self, n_users: int, n_items: int, n_factors: int, weight_init: Callable = lecun_normal_init,
                 user_shift_combination: str = "sum"):
        super().__init__(n_users, n_items, n_factors, weight_init)
        self.shift_embeddings = nn.Embedding(10, n_factors)
        weight_init(self.shift_embeddings.weight)

        self._user_shift_combination = user_shift_combination

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, shift_idx_list: torch.Tensor) -> torch.Tensor:
        user_embeddings  = self.user_embeddings(user_ids.long())
        shift_embeddings = self.shift_embeddings(shift_idx_list.long())

        if self._user_shift_combination == "sum":
            user_shift_embeddings = user_embeddings + shift_embeddings
        elif self._user_shift_combination == "multiply":
            user_shift_embeddings = user_embeddings * shift_embeddings
        else:
            raise ValueError("Unknown user_shift_combination")

        output = (user_shift_embeddings * self.item_embeddings(item_ids.long())).sum(1)
        return torch.sigmoid(output)

class MatrixFactorizationWithShiftTime(UserAndItemEmbedding):

    def __init__(self, n_users: int, n_items: int, n_factors: int, 
                    weight_init: Callable = lecun_normal_init,
                    binary: bool = True, 
                    dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout,
                    dropout_prob: float = None,
                    user_shift_combination: str = "sum"):

        super().__init__(n_users, n_items, n_factors, weight_init)
        self.shift_embeddings        = nn.Embedding(10, n_factors)
        self.weekday_embeddings      = nn.Embedding(7, n_factors)
        self.binary                  = binary
        self.dropout_module          = dropout_module
        self.dropout_prob            = dropout_prob
        self._user_shift_combination = user_shift_combination

        if dropout_prob:
            self.dropout: nn.Module = dropout_module(dropout_prob)


        hiddenSize = 100
        self.fc1  = nn.Linear(1, hiddenSize, bias=False)
        self.fc2  = nn.Linear(hiddenSize * 2, 1)

        hiddenSize = 100
        self.fc_week_1  = nn.Linear(1, hiddenSize, bias=False)
        self.fc_week_2  = nn.Linear(hiddenSize * 2, 1)

        weight_init(self.shift_embeddings.weight)
        weight_init(self.weekday_embeddings.weight)

    def shift_transform(self, x):
        shift_x1  = torch.sin(self.fc1(x))
        shift_x2  = torch.cos(self.fc1(x))
        shift_t   = self.fc2(torch.cat([shift_x1, shift_x2], 1))

        return shift_t


    def weekday_transform(self, x):
        weekday_x1  = torch.sin(self.fc_week_1(x))
        weekday_x2  = torch.cos(self.fc_week_1(x))
        weekday_t   = self.fc_week_2(torch.cat([weekday_x1, weekday_x2], 1))

        return weekday_t

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, mode_shift_idx: torch.Tensor, mode_day: torch.Tensor) -> torch.Tensor:
        user_embeddings     = self.user_embeddings(user_ids)
        item_embeddings     = self.item_embeddings(item_ids)
        shift_embeddings    = self.shift_embeddings(mode_shift_idx)
        weekday_embeddings  = self.weekday_embeddings(mode_day)

        #weekday_embeddings = self.weekday_transform(mode_day.float().view(mode_day.size()[0],-1)) 
        #shift_embeddings   = self.shift_transform(mode_shift_idx.float().view(mode_shift_idx.size()[0],-1)) 

        if self._user_shift_combination == "sum":
            user_shift_embeddings = user_embeddings + shift_embeddings + weekday_embeddings
        elif self._user_shift_combination == "multiply":
            user_shift_embeddings = user_embeddings * shift_embeddings + weekday_embeddings
        else:
            raise ValueError("Unknown user_shift_combination")



        #shift_x1  = torch.sin(self.fc1(shift_embeddings))
        #shift_x2  = torch.cos(self.fc1(shift_embeddings))
        #shift_x1  = torch.sin(self.fc1(shift_idx_list.float().view(shift_idx_list.size()[0],-1)))
        #shift_x2  = torch.cos(self.fc1(shift_idx_list.float().view(shift_idx_list.size()[0],-1)))

        #raise(Exception("{} {} {} {}".format(shift_idx_list.float(), shift_idx_list.float().shape, shift_x1, shift_x1.shape)))

        # shift_embeddings = self.shift_transform(shift_idx_list.float().view(shift_idx_list.size()[0],-1)) 
        # #self.fc2(torch.cat([shift_x1, shift_x2], 1))
        
        # if self._user_shift_combination == "sum":
        #    #item_shift_embeddings = item_embeddings + shift_embeddings
        #    user_shift_embeddings = user_embeddings + shift_embeddings           
        # elif self._user_shift_combination == "multiply":
        #    #item_shift_embeddings = item_embeddings * shift_embeddings
        #    user_shift_embeddings = user_embeddings * shift_embeddings           
        # else:
        #    raise ValueError("Unknown user_shift_combination")
        #print(shift_idx_list)
        #print(shift_idx_list.shape)

        output = (user_shift_embeddings * item_embeddings).sum(1)

        #raise(Exception("{} {} {} {} {}".format(shift_idx_list.float(), shift_idx_list.float().shape, shift_out, user_embeddings.shape, output.shape)))

        if self.binary:
            output = torch.sigmoid(output)

        return output
