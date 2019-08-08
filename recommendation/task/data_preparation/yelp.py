import json
import os
import zipfile

import luigi
import pandas as pd
import csv

from recommendation.task.data_preparation.base import BasePrepareDataFrames

BASE_DIR: str = os.path.join("output", "yelp")
DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")


class DownloadYelpDataset(luigi.Task):
    def output(self):
        return luigi.LocalTarget(os.path.join(BASE_DIR, "yelp-dataset.zip"))

    def run(self):
        raise NotImplementedError(f"Ainda não implementado. Deve-se baixar o dataset de https://www.kaggle.com/yelp-dataset/yelp-dataset e salvar o yelp-dataset em {self.output().path}")


class UnzipYelpDataset(luigi.Task):
    def requires(self):
        return DownloadYelpDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join(BASE_DIR, "yelp-dataset"))

    def run(self):
        with zipfile.ZipFile(self.input().path, "r") as zip_ref:
            zip_ref.extractall(self.output().path)


class ConvertYelpReviewsToCsvAndRemoveText(luigi.Task):
    def requires(self):
        return UnzipYelpDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "filter_reviews.csv"))

    def run(self):
        os.makedirs(DATASET_DIR, exist_ok=True)

        with open(self.output().path, "w") as csvfile:
            fieldnames = ["review_id", "user_id", "business_id", "stars", "useful", "funny", "cool", "date"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            with open(os.path.join(self.input().path, "review.json"), "r") as json_file:
                for review_line in json_file:
                    review = json.loads(review_line)
                    del review["text"]
                    writer.writerow(review)


class ConvertYelpBusinessesToCsv(luigi.Task):
    def requires(self):
        return UnzipYelpDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "filter_businesses.csv"))

    def run(self):
        os.makedirs(DATASET_DIR, exist_ok=True)

        with open(self.output().path, "w") as csvfile:
            fieldnames = ["business_id", "name", "address", "city", "state", "postal_code", "latitude", "longitude",
                          "stars", "review_count", "is_open", "attributes", "categories", "hours"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            with open(os.path.join(self.input().path, "business.json"), "r") as json_file:
                for business_line in json_file:
                    review = json.loads(business_line)
                    writer.writerow(review)


class IndexUsersAndBusinessesOfYelpReviews(luigi.Task):
    filter_restaurants: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return ConvertYelpReviewsToCsvAndRemoveText(), ConvertYelpBusinessesToCsv()

    def output(self):
        if self.filter_restaurants:
            return luigi.LocalTarget(os.path.join(DATASET_DIR, "indexed_reviews_for_restaurants.csv")), \
                   luigi.LocalTarget(os.path.join(DATASET_DIR, "user_indices_for_restaurants.csv")), \
                   luigi.LocalTarget(os.path.join(DATASET_DIR, "business_indices_for_restaurants.csv"))
        else:
            return luigi.LocalTarget(os.path.join(DATASET_DIR, "indexed_reviews.csv")), \
                   luigi.LocalTarget(os.path.join(DATASET_DIR, "user_indices.csv")), \
                   luigi.LocalTarget(os.path.join(DATASET_DIR, "business_indices.csv"))

    def run(self):
        df = pd.read_csv(self.input()[0].path)

        if self.filter_restaurants:
            business_df = pd.read_csv(self.input()[1].path)
            restaurant_df = business_df[(~business_df.categories.isna()) &
                                        (business_df.categories.str.contains("Restaurant"))]
            restaurant_ids = set(restaurant_df["business_id"].values)

            df = df[df["business_id"].isin(restaurant_ids)]

        df["user_id"] = df["user_id"].astype("category")
        df["business_id"] = df["business_id"].astype("category")
        df["user_idx"] = df["user_id"].cat.codes
        df["business_idx"] = df["business_id"].cat.codes

        df.to_csv(self.output()[0].path, index=False)

        pd.DataFrame(data=enumerate(df["user_id"].cat.categories), columns=["user_idx", "user_id"])\
            .to_csv(self.output()[1].path, index=False)
        pd.DataFrame(data=enumerate(df["business_id"].cat.categories), columns=["business_idx", "business_id"]) \
            .to_csv(self.output()[2].path, index=False)


class PrepareYelpRatingsDataFrames(BasePrepareDataFrames):
    filter_restaurants: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return IndexUsersAndBusinessesOfYelpReviews(filter_restaurants=self.filter_restaurants)

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def stratification_property(self) -> str:
        return None

    @property
    def num_users(self):
        if not hasattr(self, "_num_users"):
            user_indices_df = pd.read_csv(self.input()[1].path)
            self._num_users = len(user_indices_df)
        return self._num_users

    @property
    def num_businesses(self):
        if not hasattr(self, "_num_businesses"):
            business_indices_df = pd.read_csv(self.input()[2].path)
            self._num_businesses = len(business_indices_df)
        return self._num_businesses

    def read_data_frame(self) -> pd.DataFrame:
        df = pd.read_csv(self.input()[0].path)
        df["n_users"] = self.num_users
        df["n_items"] = self.num_businesses

        return df


class PrepareYelpAllUserRatingsDataFrames(BasePrepareDataFrames):
    filter_restaurants: bool = luigi.BoolParameter(default=True)

    def requires(self):
        return IndexUsersAndBusinessesOfYelpReviews(filter_restaurants=self.filter_restaurants)

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def stratification_property(self) -> str:
        return None

    def read_data_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.input()[0].path)

    @property
    def num_users(self):
        if not hasattr(self, "_num_users"):
            user_indices_df = pd.read_csv(self.input()[1].path)
            self._num_users = len(user_indices_df)
        return self._num_users

    @property
    def num_businesses(self):
        if not hasattr(self, "_num_businesses"):
            business_indices_df = pd.read_csv(self.input()[2].path)
            self._num_businesses = len(business_indices_df)
        return self._num_businesses

    def transform_data_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[["user_idx", "business_idx", "stars"]]
        df = df.groupby('user_idx')[['business_idx', 'stars']].apply(lambda x: x.values.tolist()).reset_index()
        df.columns = ["user_idx", "stars_per_business"]

        df["n_users"] = self.num_users
        df["n_items"] = self.num_businesses

        return df


class PrepareYelpAllBusinessRatingsDataFrames(BasePrepareDataFrames):
    filter_restaurants: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return IndexUsersAndBusinessesOfYelpReviews(filter_restaurants=self.filter_restaurants)

    @property
    def dataset_dir(self) -> str:
        return DATASET_DIR

    @property
    def stratification_property(self) -> str:
        return None

    def read_data_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.input()[0].path)

    @property
    def num_users(self):
        if not hasattr(self, "_num_users"):
            user_indices_df = pd.read_csv(self.input()[1].path)
            self._num_users = len(user_indices_df)
        return self._num_users

    @property
    def num_businesses(self):
        if not hasattr(self, "_num_businesses"):
            business_indices_df = pd.read_csv(self.input()[2].path)
            self._num_businesses = len(business_indices_df)
        return self._num_businesses

    def transform_data_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[["business_idx", "user_idx", "stars"]]
        df = df.groupby('business_idx')[['user_idx', 'stars']].apply(lambda x: x.values.tolist()).reset_index()
        df.columns = ["business_idx", "stars_per_user"]

        df["n_users"] = self.num_users
        df["n_items"] = self.num_businesses

        return df
