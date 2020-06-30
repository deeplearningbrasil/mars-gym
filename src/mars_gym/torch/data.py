from typing import List

import torch
from torch.utils.data import DataLoader, Sampler, Dataset


class FasterBatchSampler(Sampler):
    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False,
    ):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    @property
    def num_samples(self):
        if not hasattr(self, "_num_samples"):
            self._num_samples = len(self.data_source)
        return self._num_samples

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            iter_list: List[int] = torch.randperm(self.num_samples).tolist()
        else:
            iter_list: List[int] = list(range(self.num_samples))
        for i in range(0, self.num_samples, self.batch_size):
            last_idx = i + self.batch_size
            if last_idx < self.num_samples or not self.drop_last:
                yield iter_list[i:last_idx]


class NoAutoCollationDataLoader(DataLoader):
    @property
    def _auto_collation(self):
        return False

    @property
    def _index_sampler(self):
        return self.batch_sampler
