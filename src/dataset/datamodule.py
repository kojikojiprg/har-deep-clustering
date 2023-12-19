from types import SimpleNamespace
from typing import Union

import numpy as np
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset

from .collective_activity import CollectiveActivityDataset
from .volleyball import VolleyballDataset
from .video import VideoDataset


class Datamodule(LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        dataset_type: str,
        cfg: SimpleNamespace,
        stage: str,
    ):
        super().__init__()
        self._dataset_type = dataset_type
        self._batch_size = cfg.batch_size

        self._dataset: Union[CollectiveActivityDataset, VolleyballDataset]
        self._val_dataset: Union[CollectiveActivityDataset, VolleyballDataset, Subset]
        if dataset_type == "collective":
            self._dataset = CollectiveActivityDataset(dataset_dir, cfg, stage)
            if stage == "train":
                self._val_dataset = Subset(
                    self._dataset, np.random.randint(0, len(self._dataset), 10).tolist()
                )
            # if stage == "train":
            #     self._val_dataset = CollectiveActivityDataset(
            #         dataset_dir, cfg, "validation"
            #     )
        elif dataset_type == "volleyball":
            self._dataset = VolleyballDataset(dataset_dir, cfg, stage)
            if stage == "train":
                self._val_dataset = Subset(
                    self._dataset, np.random.randint(0, len(self._dataset), 10).tolist()
                )
            # if stage == "train":
            #     self._val_dataset = VolleyballDataset(
            #         dataset_dir, seq_len, resize_ratio, "validation"
            #     )
        elif dataset_type == "video":
            self._dataset = VideoDataset(dataset_dir, cfg, stage)
            if stage == "train":
                self._val_dataset = Subset(
                    self._dataset, np.random.randint(0, len(self._dataset), 10).tolist()
                )
        else:
            raise KeyError

    @property
    def n_samples(self):
        return self._dataset.n_samples

    @property
    def n_samples_batch(self):
        return self._dataset.n_samples_batch

    @property
    def dataset(self):
        return self._dataset

    def train_dataloader(self):
        return DataLoader(self._dataset, self._batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset, self._batch_size, shuffle=False, num_workers=8
        )
        # return DataLoader(
        #     self._val_dataset, self._batch_size, shuffle=False, num_workers=8
        # )

    def test_dataloader(self):
        return DataLoader(self._dataset, shuffle=False, num_workers=8)

    def predict_dataloader(self):
        return DataLoader(self._dataset, shuffle=False, num_workers=8)
