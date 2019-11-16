import luigi
import torch.nn as nn
import os
import numpy as np
from typing import Callable, List, Type, Union

from recommendation.task.model.base import BaseTorchModelTraining, TORCH_ACTIVATION_FUNCTIONS, TORCH_WEIGHT_INIT, \
    TORCH_DROPOUT_MODULES
from recommendation.model.triplet_net import TripletNet, TripletNetContent, TripletNetSimpleContent, TripletNetItemSimpleContent
from recommendation.task.model.base import TORCH_WEIGHT_INIT
from recommendation.task.model.embedding import UserAndItemEmbeddingTraining
from recommendation.task.ifood import GenerateContentEmbeddings


class TripletNetTraining(UserAndItemEmbeddingTraining):
    loss_function: str = luigi.ChoiceParameter(choices=["triplet_margin", "bpr_triplet", "weighted_triplet"], default="triplet_margin")

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

    @property
    def non_textual_input_dim(self):
        if not hasattr(self, "_non_textual_input_dim"):
            self._non_textual_input_dim = int(self.metadata_data_frame.iloc[0]["non_textual_input_dim"])
        return self._non_textual_input_dim

    @property
    def vocab_size(self):
        if not hasattr(self, "_vocab_size"):
            self._vocab_size = int(self.metadata_data_frame.iloc[0]["vocab_size"])
        return self._vocab_size

    @property
    def menu_full_text_max_words(self):
        if not hasattr(self, "_menu_full_text_max_words"):
            self._menu_full_text_max_words = int(self.metadata_data_frame.iloc[0]["menu_full_text_max_words"])
        return self._menu_full_text_max_words

    @property
    def description_max_words(self):
        if not hasattr(self, "_description_max_words"):
            self._description_max_words = int(self.metadata_data_frame.iloc[0]["description_max_words"])
        return self._description_max_words

    @property
    def category_names_max_words(self):
        if not hasattr(self, "_category_names_max_words"):
            self._category_names_max_words = int(self.metadata_data_frame.iloc[0]["category_names_max_words"])
        return self._category_names_max_words

    @property
    def name_max_words(self):
        if not hasattr(self, "_name_max_words"):
            self._name_max_words = int(self.metadata_data_frame.iloc[0]["trading_name_max_words"])
        return self._name_max_words

    def create_module(self) -> nn.Module:
        input_dim: int = self.non_textual_input_dim
        vocab_size: int = self.vocab_size
        menu_full_text_max_words: int = self.menu_full_text_max_words
        description_text_max_words: int = self.description_max_words
        category_text_max_words: int = self.category_names_max_words
        name_max_words: int = self.name_max_words


        return TripletNetContent(
            input_dim=input_dim,
            vocab_size=vocab_size,
            word_embeddings_size=self.word_embeddings_size,
            word_embeddings_output=self.word_embeddings_output,
            n_users=self.n_users,
            max_text_len_description=description_text_max_words,
            max_text_len_category=category_text_max_words,
            max_text_len_name=name_max_words,
            dropout_prob=self.dropout_prob,
            dropout_module=TORCH_DROPOUT_MODULES[self.dropout_module],
            activation_function=TORCH_ACTIVATION_FUNCTIONS[self.activation_function],
            recurrence_hidden_size=self.recurrence_hidden_size,
            content_layers=self.content_layers,
            n_factors=self.n_factors,
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
        )

    def after_fit(self):
        if self.save_user_embedding_tsv or self.save_item_embedding_tsv:
            module: TripletNetContent = self.get_trained_module()
            if self.save_user_embedding_tsv:
                user_embeddings: np.ndarray = module.user_embeddings.weight.data.cpu().numpy()
                np.savetxt(os.path.join(self.output().path, "user_embeddings.tsv"), user_embeddings, delimiter="\t")
            if self.save_item_embedding_tsv:
                self._generate_content_embeddings()

    def _generate_content_embeddings(self):
        GenerateContentEmbeddings(model_module="recommendation.task.ifood", 
                                    model_cls="TripletNetContentTraining",
                                        model_task_id=self.task_id).run()

class TripletNetSimpleContentTraining(TripletNetContentTraining):
    num_filters: int = luigi.IntParameter(default=64)
    filter_sizes: List[int] = luigi.ListParameter(default=[1, 3, 5])
    binary: bool = luigi.BoolParameter(default=False)

    def create_module(self) -> nn.Module:
        input_dim: int = self.non_textual_input_dim
        vocab_size: int = self.vocab_size

        return TripletNetSimpleContent(
            input_dim=input_dim,
            vocab_size=vocab_size,
            word_embeddings_size=self.word_embeddings_size,
            n_users=self.n_users,
            num_filters=self.num_filters,
            filter_sizes=self.filter_sizes,
            dropout_prob=self.dropout_prob,
            binary=self.binary,
            dropout_module=TORCH_DROPOUT_MODULES[self.dropout_module],
            activation_function=TORCH_ACTIVATION_FUNCTIONS[self.activation_function],
            content_layers=self.content_layers,
            n_factors=self.n_factors,
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
        )

    def _generate_content_embeddings(self):
        GenerateContentEmbeddings(model_module="recommendation.task.model.triplet_net", 
                                    model_cls="TripletNetSimpleContentTraining",
                                        model_task_id=self.task_id).run()


class TripletNetItemSimpleContentTraining(TripletNetContentTraining):
    loss_function:  str = luigi.ChoiceParameter(choices=["triplet_margin", "bpr_triplet", "relative_triplet"], default="triplet_margin")
    num_filters:    int = luigi.IntParameter(default=64)
    filter_sizes: List[int] = luigi.ListParameter(default=[1, 3, 5])
    binary: bool = luigi.BoolParameter(default=False)
    use_normalize:  bool = luigi.BoolParameter(default=False)
    recurrence_hidden_size: int = luigi.IntParameter(default=40)

    def create_module(self) -> nn.Module:
        input_dim: int = self.non_textual_input_dim
        vocab_size: int = self.vocab_size
        menu_full_text_max_words: int = self.menu_full_text_max_words

        return TripletNetItemSimpleContent(
            input_dim=input_dim,
            vocab_size=vocab_size,
            word_embeddings_size=self.word_embeddings_size,
            num_filters=self.num_filters,
            filter_sizes=self.filter_sizes,
            dropout_prob=self.dropout_prob,
            use_normalize=self.use_normalize,
            content_layers=self.content_layers,

            menu_full_text_max_words=menu_full_text_max_words,
            recurrence_hidden_size=self.recurrence_hidden_size,
            binary=self.binary,
            dropout_module=TORCH_DROPOUT_MODULES[self.dropout_module],
            activation_function=TORCH_ACTIVATION_FUNCTIONS[self.activation_function],
            n_factors=self.n_factors,
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
        )

    def after_fit(self):
        if self.save_item_embedding_tsv:
            self._generate_content_embeddings()

    def _generate_content_embeddings(self):
        GenerateContentEmbeddings(model_module="recommendation.task.model.triplet_net", 
                                    model_cls="TripletNetItemSimpleContentTraining",
                                    model_task_id=self.task_id,
                                    export_tsne=True)