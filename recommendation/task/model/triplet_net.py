import luigi
import torch.nn as nn
import os
import numpy as np

from recommendation.model.triplet_net import TripletNet, TripletNetContent
from recommendation.task.model.base import TORCH_WEIGHT_INIT
from recommendation.task.model.embedding import UserAndItemEmbeddingTraining


class TripletNetTraining(UserAndItemEmbeddingTraining):
    loss_function: str = luigi.ChoiceParameter(choices=["triplet_margin", "bpr_triplet"], default="triplet_margin")

    def create_module(self) -> nn.Module:
        return TripletNet(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
        )


class TripletNetContentTraining(BaseTorchModelTraining):
    loss_function: str = luigi.ChoiceParameter(choices=["triplet_margin", "bpr_triplet"], default="triplet_margin")
    n_factors: int = luigi.IntParameter(default=128)
    weight_init: str = luigi.ChoiceParameter(choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal")
    save_user_embedding_tsv: bool = luigi.BoolParameter(default=False)
    save_item_embedding_tsv: bool = luigi.BoolParameter(default=False)
    word_embeddings_size: int = luigi.IntParameter(default=128)
    word_embeddings_output: int = luigi.IntParameter(default=128)
    dropout_prob: float = luigi.FloatParameter(default=None)
    dropout_module: str = luigi.ChoiceParameter(choices=TORCH_DROPOUT_MODULES.keys(), default="alpha")
    activation_function: str = luigi.ChoiceParameter(choices=TORCH_ACTIVATION_FUNCTIONS.keys(), default="selu")
    recurrence_hidden_size: int = luigi.IntParameter(default=40)
    content_layers: List[int] = luigi.ListParameter(default=[200, 128])

    menu_text_length: int = luigi.IntParameter(default=5000)
    description_text_length: int = luigi.IntParameter(default=200)
    category_text_length: int = luigi.IntParameter(default=250)
    input_dim: int = self.non_textual_input_dim
    vocab_size: int = self.vocab_size

                   
    def create_module(self) -> nn.Module:
        return TripletNetContent(
            input_dim=self.input_dim,
            vocab_size=self.vocab_size,
            n_users=self.n_users,
            max_text_len_description=self.description_text_length,
            max_text_len_category=self.category_text_length,
            dropout_prob=self.dropout_prob,
            dropout_module=self.dropout_module,
            activation_function=self.activation_function,
            recurrence_hidden_size=self.recurrence_hidden_size,
            content_layers=self.content_layers,
            n_items=self.n_items,
            n_factors=self.n_factors,
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
        )

    def after_fit(self):
        if self.save_user_embedding_tsv or self.save_item_embedding_tsv:
            module: TripletNetContent = self.get_trained_module()
            if self.save_user_embedding_tsv:
                user_embeddings: np.ndarray = module.user_embeddings.weight.data.cpu().numpy()
                np.savetxt(os.path.join(self.output().path, "user_embeddings.tsv"), user_embeddings, delimiter="\t")

            #TODO Restaurant embeddings