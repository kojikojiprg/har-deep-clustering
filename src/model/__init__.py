from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from numpy.typing import NDArray

from .autoencoder import Autoencoder
from .clustering import ClusteringModule


class DeepClusteringModel(LightningModule):
    def __init__(
        self,
        cfg: SimpleNamespace,
        n_samples: int,
        n_samples_batch: int,
        checkpoint_dir: str,
    ):
        super().__init__()
        # self.automatic_optimization = True

        self._cfg = cfg
        self._update_interval = cfg.update_interval
        self._n_samples = n_samples
        self._n_samples_batch = n_samples_batch

        self._ae = Autoencoder(cfg)
        self._cm = ClusteringModule(cfg, n_samples, n_samples_batch)

        self._lr = nn.MSELoss()
        self._lc = nn.KLDivLoss(reduction="sum")

        # self._c_hist_epoch = torch.full((n_samples,), torch.nan)
        # self._c_hist: list = []  # clustering history

        self._callbacks = [
            ModelCheckpoint(
                checkpoint_dir,
                filename=f"dcm_seq{cfg.seq_len}_lc_min",
                monitor="lc",
                mode="min",
                save_last=True,
            ),
        ]
        last_name = f"dcm_seq{cfg.seq_len}_last"
        self._callbacks[0].CHECKPOINT_NAME_LAST = last_name

    @property
    def callbacks(self) -> list:
        return self._callbacks

    @property
    def clustering_history(self) -> list:
        return self._c_hist

    @property
    def target_distribution(self) -> NDArray:
        return self._cm.target_distribution.detach().cpu().numpy()

    def training_step(self, batch, batch_idx, optimizer_idx):
        frames, flows, bboxs, batch_idxs = batch

        # get constants
        batch_size, _, seq_len = frames.shape[:3]
        w = self._cfg.fake_size.w
        h = self._cfg.fake_size.h

        # train autoencoder
        if optimizer_idx == 0:
            z, frames_out, flows_out = self._ae(frames, flows, bboxs)

            # calc loss autoencoder
            lr_total = 0
            for i in range(batch_size):
                for j in range(self._n_samples_batch):
                    try:
                        bx = bboxs[i, j].cpu().numpy()
                    except IndexError:
                        continue  # last batch of epoch
                    if np.any(np.isnan(bx)):
                        continue
                    x1, y1, x2, y2 = bx.astype(int)
                    frame_bbox = frames[i, :, seq_len // 2, y1:y2, x1:x2]
                    flow_bbox = flows[i, :, seq_len // 2, y1:y2, x1:x2]
                    frame_bbox = F.resize(frame_bbox, (w, h), antialias=True)
                    flow_bbox = F.resize(flow_bbox, (w, h), antialias=True)

                    lr_total += self._lr(frames_out[i, j], frame_bbox)
                    lr_total += self._lr(flows_out[i, j], flow_bbox)

            self.log_dict({"lr": lr_total}, prog_bar=True, on_step=True, on_epoch=True)
            if np.all(np.isnan(bboxs.cpu().numpy())):
                # all of bboxs are nan
                return None
            return lr_total

        # train clustering
        if optimizer_idx == 1:
            z, _, _ = self._ae(frames, flows, bboxs)
            z, s, c = self._cm(z.detach())

            # update clustering label
            # for i, batch_idx in enumerate(batch_idxs.cpu()):
            #     for j in range(self._n_samples_batch):
            #         idx = batch_idx * self._n_samples_batch + j
            #         self._c_hist_epoch[idx] = c[i, j]

            if (
                self.current_epoch % self._update_interval == 0  # periodical update
                or self.current_epoch == self._cfg.epochs - 1  # last epoch
            ):
                # save clustering label
                # self._c_hist.append(self._c_hist_epoch.cpu().detach().numpy())
                # update target
                self._cm.update_target_distribution(s, batch_idxs)

            # calc loss clustering
            lc_total = 0
            for i, batch_idx in enumerate(batch_idxs):
                idx = batch_idx * self._n_samples_batch
                tmp_target = self._cm.target_distribution[
                    idx : idx + self._n_samples_batch
                ]
                tmp_s = s[i]
                lc_total += self._lc(tmp_s.log(), tmp_target)
            self.log_dict({"lc": lc_total}, prog_bar=True, on_step=True, on_epoch=True)

            return lc_total

    def predict_step(self, batch, batch_idx, dataloader_idx):
        frames, flows, bboxs, batch_idxs = batch

        z, _, _, _, c = self(frames, flows, bboxs)
        preds = []
        for i in batch_idxs:
            for j in range(self._n_sampels_batch):
                data = {
                    "sample_num": i + j,
                    "z": z[i, j],
                    "c": c[i, j],
                }
                preds.append(data)

        return preds

    def configure_optimizers(self):
        optim_ae = torch.optim.Adam(self._ae.parameters(), self._cfg.optim.lr_rate_ae)
        optim_cm = torch.optim.Adam(self._cm.parameters(), self._cfg.optim.lr_rate_cm)
        step_size = self._cfg.epochs // 4
        sch_ae = torch.optim.lr_scheduler.StepLR(
            optim_ae, step_size=step_size, gamma=0.5
        )
        sch_cm = torch.optim.lr_scheduler.StepLR(
            optim_cm, step_size=step_size, gamma=0.5
        )
        return [optim_ae, optim_cm], [sch_ae, sch_cm]
