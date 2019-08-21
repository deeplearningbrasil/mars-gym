from typing import List, Callable, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from recommendation.utils import lecun_normal_init
#import ipdb; ipdb.set_trace()


# PYTHONPATH="." luigi --module recommendation.task.model.deep DeepTraining --project criteo --local-scheduler --input-d-dim 13 
# --input-c-dim 26 --generator-workers 0 --loss-function bce_loss --n-factors 10 --epochs 30
class DeepCTR(nn.Module):
    def __init__(self, input_d_dim: int, input_c_dim: int,  
                 dropout_prob: float,
                 n_factors: int,
                 size_embs: List[int],
                 dense_layers: List[int],
                 activation_function: Callable = F.selu, weight_init: Callable = lecun_normal_init,
                 dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout):
        super().__init__()

        #self.activation_function = activation_function
        self.weight_init = weight_init
        self.size_embs   = size_embs
        self.n_factors   = n_factors
        self.input_c_dim = input_c_dim

        # Def Dense Layer 
        def dense_layer(in_f, out_f, *args, **kwargs):
            return nn.Sequential(
                nn.Linear(in_f, out_f, *args, **kwargs),
                nn.ReLU()
            )   

        # Def Embedding Layer
        def embs_layer(size, n_factors):
            return nn.Embedding(size, n_factors)

        # Dense Layers
        size_layers = [input_d_dim + (n_factors * input_c_dim), *dense_layers]
        blocks      = [dense_layer(in_f, out_f) for in_f, out_f in zip(size_layers, size_layers[1:])]
        self.dense_block = nn.Sequential(*blocks)

        # Embedding Layers
        #self.embeddings = np.array([embs_layer(int(self.size_embs[i] + 1), self.n_factors) for i in range(self.input_c_dim)])
        #print(self.embeddings)
        #print(size_embs)
        #print("-----------------")

        # Logistic Loss
        self.logistic = nn.Sequential(
            nn.Linear(size_layers[-1], 1),
            nn.Sigmoid()
        )

        # self.embs = {}
        # for i in range(self.input_c_dim):
        #     self.embs[i] = embs_layer(int(self.size_embs[i] + 1), self.n_factors).to(self.torch_device())


        self.emb_c1  = embs_layer(int(self.size_embs[0] + 1), self.n_factors)
        self.emb_c2  = embs_layer(int(self.size_embs[1] + 1), self.n_factors)
        self.emb_c3  = embs_layer(int(self.size_embs[2] + 1), self.n_factors)
        self.emb_c4  = embs_layer(int(self.size_embs[3] + 1), self.n_factors)
        self.emb_c5  = embs_layer(int(self.size_embs[4] + 1), self.n_factors)
        self.emb_c6  = embs_layer(int(self.size_embs[5] + 1), self.n_factors)
        self.emb_c7  = embs_layer(int(self.size_embs[6] + 1), self.n_factors)
        self.emb_c8  = embs_layer(int(self.size_embs[7] + 1), self.n_factors)
        self.emb_c9  = embs_layer(int(self.size_embs[8] + 1), self.n_factors)
        self.emb_c10 = embs_layer(int(self.size_embs[9] + 1), self.n_factors)
        self.emb_c11 = embs_layer(int(self.size_embs[10] + 1), self.n_factors)
        self.emb_c12 = embs_layer(int(self.size_embs[11] + 1), self.n_factors)
        self.emb_c13 = embs_layer(int(self.size_embs[12] + 1), self.n_factors)
        self.emb_c14 = embs_layer(int(self.size_embs[13] + 1), self.n_factors)
        self.emb_c15 = embs_layer(int(self.size_embs[14] + 1), self.n_factors)
        self.emb_c16 = embs_layer(int(self.size_embs[15] + 1), self.n_factors)
        self.emb_c17 = embs_layer(int(self.size_embs[16] + 1), self.n_factors)
        self.emb_c18 = embs_layer(int(self.size_embs[17] + 1), self.n_factors)
        self.emb_c19 = embs_layer(int(self.size_embs[18] + 1), self.n_factors)
        self.emb_c20 = embs_layer(int(self.size_embs[19] + 1), self.n_factors)
        self.emb_c21 = embs_layer(int(self.size_embs[20] + 1), self.n_factors)
        self.emb_c22 = embs_layer(int(self.size_embs[21] + 1), self.n_factors)
        self.emb_c23 = embs_layer(int(self.size_embs[22] + 1), self.n_factors)
        self.emb_c24 = embs_layer(int(self.size_embs[23] + 1), self.n_factors)
        self.emb_c25 = embs_layer(int(self.size_embs[24] + 1), self.n_factors)
        self.emb_c26 = embs_layer(int(self.size_embs[25] + 1), self.n_factors)

        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def torch_device(self):
        return torch.device(f"cuda:0")
    
    def forward(self, xd: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        # for i, emb in enumerate(self.embeddings):
        #     print(x.shape, emb(xc[:, i]).shape)
        #     x  = torch.cat((x, emb(xc[:, i])), 1)
        
        #embs = [emb(xc[:, i]) for i, emb in enumerate(self.embeddings)]
        #x  = torch.stack(embs, 1)
        #x  = torch.cat(embs, 1)

        xc1  = self.emb_c1(xc[:, 0])
        xc2  = self.emb_c2(xc[:, 1])
        xc3  = self.emb_c3(xc[:, 2])
        xc4  = self.emb_c4(xc[:, 3])
        xc5  = self.emb_c5(xc[:, 4])
        xc6  = self.emb_c6(xc[:, 5])
        xc7  = self.emb_c7(xc[:, 6])
        xc8  = self.emb_c8(xc[:, 7])
        xc9  = self.emb_c9(xc[:, 8])
        xc10 = self.emb_c10(xc[:, 9])
        xc11 = self.emb_c11(xc[:, 10])
        xc12 = self.emb_c12(xc[:, 11])
        xc13 = self.emb_c13(xc[:, 12])
        xc14 = self.emb_c14(xc[:, 13])
        xc15 = self.emb_c15(xc[:, 14])
        xc16 = self.emb_c16(xc[:, 15])
        xc17 = self.emb_c17(xc[:, 16])
        xc18 = self.emb_c18(xc[:, 17])
        xc19 = self.emb_c19(xc[:, 18])
        xc20 = self.emb_c20(xc[:, 19])
        xc21 = self.emb_c21(xc[:, 20])
        xc22 = self.emb_c22(xc[:, 21])
        xc23 = self.emb_c23(xc[:, 22])
        xc24 = self.emb_c24(xc[:, 23])
        xc25 = self.emb_c25(xc[:, 24])
        xc26 = self.emb_c26(xc[:, 25])

        x = torch.cat([xd.float(), xc1, xc2, xc3, xc4, xc5, xc6, xc7, xc8, xc9, xc10, xc11, xc12,
            xc13, xc14, xc15, xc16, xc17, xc18, xc19, xc20, xc21, xc22, xc23, xc24, xc25, xc26], 1)

        # concat = [xd.float()]
        # for i, emb in self.embs.items():
        #     concat.append(emb(xc[:, i]))
        #     #print(x.shape, emb(xc[:, i]).shape)
        #     #x  = torch.cat((x, emb(xc[:, i])), 1)

        #x = x.view(-1)
        #x = self.layers(x)

        x = self.dense_block(x)
        x = self.logistic(x)

        return x
