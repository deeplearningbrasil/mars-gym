from typing import List
import torch
import torch.nn as nn

import luigi

from recommendation.model.deep import DeepCTR
from recommendation.task.meta_config import RecommenderType
from recommendation.task.model.base import BaseTorchModelTraining, TORCH_ACTIVATION_FUNCTIONS, TORCH_WEIGHT_INIT, \
    TORCH_DROPOUT_MODULES
from recommendation.torch import MaskedZeroesLoss
import pandas as pd
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
tqdm.pandas()
from recommendation.utils import chunks
import math
import numpy as np
import os

class DeepTraining(BaseTorchModelTraining):
    input_d_dim: int = luigi.IntParameter(default=100)
    input_c_dim: int = luigi.IntParameter(default=100)
    n_factors:   int = luigi.IntParameter(default=1)

    dense_layers: List[int] = luigi.ListParameter(default=[1024, 512, 256])

    dropout_prob: float = luigi.FloatParameter(default=None)
    activation_function: str = luigi.ChoiceParameter(choices=TORCH_ACTIVATION_FUNCTIONS.keys(), default="relu")
    weight_init: str = luigi.ChoiceParameter(choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal")
    dropout_module: str = luigi.ChoiceParameter(choices=TORCH_DROPOUT_MODULES.keys(), default="alpha")


    def create_module(self) -> nn.Module:
        return DeepCTR(input_d_dim=self.input_d_dim, input_c_dim=self.input_c_dim, 
                       n_factors=self.n_factors,
                       size_embs=list(self.size_embs.values()),
                       dense_layers=self.dense_layers,
                       dropout_prob=self.dropout_prob,
                       activation_function=TORCH_ACTIVATION_FUNCTIONS[self.activation_function],
                       weight_init=TORCH_WEIGHT_INIT[self.weight_init],
                       dropout_module=TORCH_DROPOUT_MODULES[self.dropout_module])

    @property
    def size_embs(self):
        columns = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 
                    'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 
                    'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 
                    'C23', 'C24', 'C25', 'C26']

        if not hasattr(self, "_size_embs"):
            self._size_embs = self.build_n_categories(columns)
        return self._size_embs

            
    def build_n_categories(self, columns):
        train_df = pd.read_csv(self.input()[0].path)
        val_df   = pd.read_csv(self.input()[1].path)

        df       = pd.concat([train_df, val_df])

        unique_vals = {}

        for c in columns:
            unique_vals[c] = len(df[c].unique())

        return unique_vals

    def after_fit(self):
        module: DeepCTR = self.get_trained_module()
        self.generate_submission_file(model=module)

    # Id,Predicted
    # 60000000,0.361366158048
    # 60000001,0.619944481451
    def generate_submission_file(self, model: DeepCTR = None):
        if model == None:
            model: DeepCTR = self.get_trained_module()

        df_test    = self.test_dataset._data_frame
        scores: List[float] = []

        # Interator per batch
        for indices in tqdm(chunks(range(len(df_test)), self.batch_size),
                            total=math.ceil(len(df_test) / self.batch_size)):
            rows: pd.DataFrame = df_test.iloc[indices]
                
            dense_columns = torch.tensor(rows[self.test_dataset._dense_columns].values,       dtype=torch.float64)
            cat_columns   = torch.tensor(rows[self.test_dataset._categorical_columns].values, dtype=torch.int64)
            batch_scores: torch.Tensor = model(dense_columns.to(self.torch_device),
                                                cat_columns.to(self.torch_device))
            scores.extend(batch_scores.detach().cpu().numpy().tolist())
            
        # Save csv
        df = pd.DataFrame({'Predicted':  np.reshape(scores,-1)})
        df['Id'] = [60000000+(x) for x in range(len(df))]
        df.to_csv(os.path.join(self.output().path, "submission.csv"), index=False)

    @property
    def test_dataset(self) -> Dataset:
        if not hasattr(self, "_test_dataset"):
            test_df = pd.read_csv(self.input()[2].path)
            self._test_dataset = self.project_config.dataset_class(test_df, self.project_config)
            self._test_dataset._train = False
        return self._test_dataset