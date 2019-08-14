import abc
import os

import luigi
import numpy as np

from recommendation.model.embedding import UserAndItemEmbedding
from recommendation.task.model.base import BaseTorchModelTraining, TORCH_WEIGHT_INIT


class UserAndItemEmbeddingTraining(BaseTorchModelTraining, metaclass=abc.ABCMeta):
    n_factors: int = luigi.IntParameter(default=20)
    weight_init: str = luigi.ChoiceParameter(choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal")
    save_user_embedding_tsv: bool = luigi.BoolParameter(default=False)
    save_item_embedding_tsv: bool = luigi.BoolParameter(default=False)

    def after_fit(self):
        if self.save_user_embedding_tsv or self.save_item_embedding_tsv:
            module: UserAndItemEmbedding = self.get_trained_module()
            if self.save_user_embedding_tsv:
                user_embeddings: np.ndarray = module.user_embeddings.weight.data.cpu().numpy()
                np.savetxt(os.path.join(self.output().path, "user_embeddings.tsv"), user_embeddings, delimiter="\t")
            if self.save_item_embedding_tsv:
                item_embeddings: np.ndarray = module.item_embeddings.weight.data.cpu().numpy()
                np.savetxt(os.path.join(self.output().path, "item_embeddings.tsv"), item_embeddings, delimiter="\t")