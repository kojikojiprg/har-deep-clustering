from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset


class AbstractDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, seq_len: int):
        super().__init__()
        self._default_float_dtype = torch.get_default_dtype()

        self._seq_len = seq_len
        self._frames: list = []
        self._flows: list = []
        self._bboxs: list = []
        self._n_samples_batch = 0

    def transform_imgs(self, imgs):
        imgs = torch.tensor(np.array(imgs).transpose((0, 3, 1, 2))).contiguous()
        if isinstance(imgs, torch.ByteTensor):
            imgs = (imgs / 255.0) * 2 - 1
        return imgs.to(dtype=self._default_float_dtype)

    @property
    def n_samples(self):
        return len(self) * self._n_samples_batch

    @property
    def n_samples_batch(self):
        return self._n_samples_batch

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError
