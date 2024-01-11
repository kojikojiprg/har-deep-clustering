import os
from abc import ABCMeta, abstractmethod
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from numpy.typing import NDArray

from .clustering_module import ClusteringModule


class AbstractDeepClusteringModel(LightningModule, metaclass=ABCMeta):
    def __init__(
        self,
        model_type: str,
        cfg: SimpleNamespace,
        n_samples: int,
        n_samples_batch: int,
        checkpoint_dir: Optional[str] = None,
        version: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.update_interval = cfg.update_interval
        self.n_samples = n_samples
        self.n_samples_batch = n_samples_batch
        self.lmd_cm = cfg.optim.lmd_cm

        self.cm = ClusteringModule(cfg, n_samples, n_samples_batch)

        self.lr = nn.MSELoss()
        self.lc = nn.KLDivLoss(reduction="sum")

        if checkpoint_dir is not None:
            checkpoint_dir = os.path.join(checkpoint_dir, model_type)
            if version is not None:
                vstr = f"-v{version}"
            else:
                vstr = "-v0"
            self._callbacks = [
                ModelCheckpoint(
                    checkpoint_dir,
                    filename=f"dcm_seq{cfg.seq_len}_lc_min{vstr}",
                    monitor="lc",
                    mode="min",
                    save_last=True,
                ),
            ]
            last_name = f"dcm_seq{cfg.seq_len}_last{vstr}"
            self._callbacks[0].CHECKPOINT_NAME_LAST = last_name

    @property
    def callbacks(self) -> list:
        return self._callbacks

    @property
    def target_distribution(self) -> NDArray:
        return self.cm.target_distribution.detach().cpu().numpy()

    @property
    def centroids(self) -> NDArray:
        return np.array([c.detach().numpy() for c in iter(self.cm.centroids.cpu())])

    def calc_lc(self, s, bboxs, data_idxs):
        # calc clustering loss
        lc_total = 0
        # if self.current_epoch + 1 >= self._cfg.clustering_start_epoch:
        for b, data_idx in enumerate(data_idxs):
            ti = data_idx * self.n_samples_batch
            tmp_t = self.cm.target_distribution[ti : ti + self.n_samples_batch]
            tmp_s = s[b]

            bboxs_batch = bboxs[b].cpu().numpy()
            mask_not_nan = ~np.isnan(bboxs_batch).any(axis=1)
            tmp_t = tmp_t[mask_not_nan]
            tmp_s = tmp_s[mask_not_nan]
            lc_total = lc_total + self.lc(tmp_s.log(), tmp_t.detach())

        return lc_total

    @abstractmethod
    def training_step(self, batch, batch_idx, optimizer_idx):
        raise NotImplementedError

    # @abstractmethod
    # def validation_step(self, batch, batch_idx):
    #     raise NotImplementedError

    @abstractmethod
    def predict_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError
