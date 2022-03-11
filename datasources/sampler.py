import random
import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from typing import Iterator, List


def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)


class BatchSampler(Sampler):
    """
    sampler to yield batches of same participant data

    """

    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool, shuffle: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.total_part = list(set(list(self.sampler.keys())))  # getting unique participant names
        self.length = 0
        for k, v in self.sampler.items():
            self.length += len(v)

        for k, v in self.sampler.items():
            np.random.shuffle(self.sampler[k])

    def make_batches(self):
        all_batches = []
        for idd, all_indices_id in self.sampler.items():
            batches = chunk(all_indices_id, self.batch_size)
            all_batches += list(batches)
        return all_batches

    def __iter__(self) -> Iterator[List[int]]:
        combined_batches = self.make_batches()
        if self.drop_last:
            combined = [batch.tolist() for batch in combined_batches if len(batch)==self.batch_size]
        else:
            combined = [batch.tolist() for batch in combined_batches]
        random.shuffle(combined)
        return iter(combined)

    def __len__(self) -> int:
        if self.drop_last:
            return self.length // self.batch_size  # type: ignore[arg-type]
        else:
            return (self.length + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]



