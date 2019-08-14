from typing import Tuple

import torch

from recommendation.model.embedding import UserAndItemEmbedding


class TripletNet(UserAndItemEmbedding):
    def forward(self, user_ids: torch.Tensor, positive_item_ids: torch.Tensor,
                negative_item_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_embeddings(user_ids), self.item_embeddings(positive_item_ids), \
               self.item_embeddings(negative_item_ids)
