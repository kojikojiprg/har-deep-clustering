from abc import ABCMeta

import numpy as np
import torch
from torch.utils.data import Dataset


class AbstractDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, seq_len: int, resize_ratio: float):
        super().__init__()
        self._default_float_dtype = torch.get_default_dtype()

        self._seq_len = seq_len
        self._resize_ratio = resize_ratio
        self._frames: list = []
        self._flows: list = []
        self._bboxs: list = []
        self._idx_ranges = None
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

    def __len__(self):
        return self._idx_ranges[-1, 1]

    def __getitem__(self, idx):
        video_idx = np.where(
            (self._idx_ranges[:, 0] <= idx) & (idx < self._idx_ranges[:, 1])
        )[0].item()
        data_idx = int(idx - self._idx_ranges[video_idx, 0])

        frames = self._frames[video_idx][data_idx : data_idx + self._seq_len].transpose(1, 0)
        flows = self._flows[video_idx][data_idx : data_idx + self._seq_len].transpose(1, 0)
        bboxs = self._bboxs[video_idx][data_idx + self._seq_len]
        # append dmy bboxs
        if len(bboxs) < self._n_samples_batch:
            diff_num = self._n_samples_batch - len(bboxs)
            dmy_bboxs = [np.full((4,), np.nan) for _ in range(diff_num)]
            bboxs = np.append(bboxs, dmy_bboxs, axis=0)
        bboxs = torch.Tensor(bboxs)

        return frames, flows, bboxs, idx
