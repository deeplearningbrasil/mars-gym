# Criteo AI Lab
#
# https://www.kaggle.com/c/criteo-display-ad-challenge
# https://ailab.criteo.com/kaggle-contest-dataset-now-available-academic-use/
#

import json
import os
import zipfile
from typing import Tuple, List, Iterator, Union, Sized, Callable

import luigi
import pandas as pd
import csv
import category_encoders as ce

from recommendation.task.data_preparation.base import BasePrepareDataFrames
from sklearn.model_selection import train_test_split

BASE_DIR: str = os.path.join("output", "criteo")
DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")

TRAIN_FILE: str  = "train.txt"
TEST_FILE: str   = "test.txt"

class DownloadYelpDataset(luigi.Task):
    def output(self):
        return luigi.LocalTarget(os.path.join(BASE_DIR, "yelp-dataset.zip"))

    def run(self):
        raise NotImplementedError(f"Ainda não implementado. Deve-se baixar o dataset de https://www.kaggle.com/yelp-dataset/yelp-dataset e salvar o yelp-dataset em {self.output().path}")


DATASET_COLUMNS = ['TARGET', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13',
                    'C1', 'C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17',
                    'C18','C19','C20','C21','C22','C23','C24','C25','C26']

class PrepareDataFrames(BasePrepareDataFrames):
    test_size: float = luigi.FloatParameter(default=0.2)
    seed: int = luigi.IntParameter(default=42)
    no_cross_columns: bool = luigi.BoolParameter(default=False)

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def stratification_property(self) -> str:
        return "TARGET"

    def read_data_frame(self) -> pd.DataFrame:
        pass

    # def requires(self):
    #    return None

    def output(self) -> Tuple[luigi.LocalTarget, luigi.LocalTarget, luigi.LocalTarget]:
        task_hash = self.task_id.split("_")[-1]
        return (luigi.LocalTarget(os.path.join(DATASET_DIR,
                                               "train_%.2f_%d_%d_%s.csv" % (
                                                   self.test_size, self.seed, self.no_cross_columns, task_hash))),
                luigi.LocalTarget(os.path.join(DATASET_DIR, 
                                                "val_%.2f_%d_%d_%s.csv" % (
                                                    self.test_size, self.seed, self.no_cross_columns, task_hash))),
                luigi.LocalTarget(os.path.join(DATASET_DIR, 
                                                "test_%.2f_%d_%d_%s.csv" % (
                                                    self.test_size, self.seed, self.no_cross_columns, task_hash))))

    def run(self):
        # Train Dataset
        train_df = pd.read_csv(os.path.join(DATASET_DIR, TRAIN_FILE), sep='\t', header=None)
        train_df.columns   = DATASET_COLUMNS
        
        # Test Dataset
        test_df = pd.read_csv(os.path.join(DATASET_DIR, TEST_FILE), sep='\t', header=None)
        test_df.columns    = DATASET_COLUMNS[1:]
        
        # Preprocess Datasets
        train_df, test_df  = self.preprocess(train_df, test_df)

        # Sprint Val Dataset
        train_df, val_df   = train_test_split(train_df, test_size=self.test_size, random_state=self.seed)

        train_df.to_csv(self.output()[0].path, index=False)
        val_df.to_csv(self.output()[1].path, index=False)
        test_df.to_csv(self.output()[2].path, index=False)
        

    def preprocess(self, df, test_df = None):
        print(df.describe(include = ['object', 'float', 'int']))

        categorical_columns = list(df.select_dtypes(include=['object']).columns)

        #if not self.no_cross_columns:
            # my understanding on how to replicate what layers.crossed_column does. One
            # can read here: https://www.tensorflow.org/tutorials/linear.
        #    df = self.cross_columns(df, categorical_columns)
            
        # Encoder categorical Columns
        #
        df, test_df = self.encoder_categorical_columns(df, test_df)

        df      = df.fillna(-1)
        test_df = test_df.fillna(-1)

        return df, test_df

    def encoder_categorical_columns(self, df, test_df = None):
        """
        Encoder Categorical Columns
        """

        # Categorical Columns After Cross
        categorical_columns = list(df.select_dtypes(include=['object']).columns)

        # encoder = ce.OneHotEncoder(cols=list(df.select_dtypes(include=['object']).columns),
        #                             use_cat_names=True, drop_invariant=True )

        encoder = ce.OrdinalEncoder(cols=categorical_columns)
        df_t    = encoder.fit_transform(df[DATASET_COLUMNS[1:]])

        if test_df is not None:
            test_df = encoder.transform(test_df)

        df_t['TARGET'] = df['TARGET']

        return df_t, test_df

    def cross_columns(self, df, x_cols):
        """simple helper to build the crossed columns in a pandas dataframe
        """
        for c1 in x_cols:
            for c2 in x_cols:
                if c1 != c2:
                    df["{}_{}".format(c1, c2)] = df.apply(lambda row: "{}_{}".format(row[c1], row[c2]), axis=1)

        return df

