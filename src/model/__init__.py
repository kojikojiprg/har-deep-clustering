from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
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
        checkpoint_dir: Optional[str] = None,
    ):
        super().__init__()
        self.automatic_optimization = False

        self._cfg = cfg
        self._update_interval = cfg.update_interval
        self._n_accumulate = cfg.accumulate_grad_batches
        self._n_samples = n_samples
        self._n_samples_batch = n_samples_batch
        self._lmd_cm = cfg.optim.lmd_cm

        self._ae_frame = Autoencoder(cfg, 3)
        self._ae_flow = Autoencoder(cfg, 2)
        self._cm = ClusteringModule(cfg, n_samples, n_samples_batch)

        self._lr = nn.MSELoss()
        self._lc = nn.KLDivLoss(reduction="sum")

        # self._c_hist_epoch = torch.full((n_samples,), torch.nan)
        # self._c_hist: list = []  # clustering history

        if checkpoint_dir is not None:
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

    @property
    def centroids(self) -> NDArray:
        return np.array(
            [c.detach().numpy() for c in list(iter(self._cm.centroids.cpu()))]
        )

    def training_step(self, batch, batch_idx):
        frames, flows, bboxs, data_idxs = batch
        optim_ae_frame, optim_ae_flow, optim_cm = self.optimizers()
        sch_ae_frame, sch_ae_flow, sch_cm = self.lr_schedulers()

        # train autoencoder
        z_frame, fake_frames = self._ae_frame(frames)
        z_flow, fake_flows = self._ae_flow(flows)

        # calc loss autoencoder
        lr_frame = self._lr(frames[:, :, -1], fake_frames) / self._n_accumulate
        lr_flow = self._lr(flows[:, :, -1], fake_flows) / self._n_accumulate

        # step autoencoder optimizer
        self.manual_backward(lr_frame)
        if (batch_idx + 1) % self._n_accumulate == 0:
            optim_ae_frame.step()
            optim_ae_frame.zero_grad()
            sch_ae_frame.step()

        self.manual_backward(lr_flow)
        if (batch_idx + 1) % self._n_accumulate == 0:
            optim_ae_flow.step()
            optim_ae_flow.zero_grad()
            sch_ae_flow.step()

        # train clustering
        if self.current_epoch + 1 >= self._cfg.clustering_start_epoch:
            z = z_frame.detach() + z_flow.detach()
            z, s, c = self._cm(z, bboxs)

            # update clustering label
            # for i, data_idx in enumerate(data_idxs.cpu()):
            #     for j in range(self._n_samples_batch):
            #         idx = data_idx * self._n_samples_batch + j
            #         self._c_hist_epoch[idx] = c[i, j]

            if (
                self.current_epoch % self._update_interval == 0  # periodical update
                or self.current_epoch == self._cfg.epochs - 1  # last epoch
            ):
                # save clustering label
                # self._c_hist.append(self._c_hist_epoch.cpu().detach().numpy())
                # update target
                self._cm.update_target_distribution(s, data_idxs)

            # calc loss clustering
            lc_total = 0
            for i, data_idx in enumerate(data_idxs):
                idx = data_idx * self._n_samples_batch
                tmp_target = self._cm.target_distribution[
                    idx : idx + self._n_samples_batch
                ]
                tmp_s = s[i]
                lc_total = lc_total + self._lc(tmp_s.log(), tmp_target)
            lc_total = lc_total / self._n_accumulate * self._lmd_cm

            # step clustering optimizer
            self.manual_backward(lc_total)
            if (batch_idx + 1) % self._n_accumulate == 0:
                optim_cm.step()
                optim_cm.zero_grad()
                sch_cm.step()
        else:
            lc_total = torch.nan

        self.log_dict(
            {"lr_frame": lr_frame, "lr_flow": lr_flow, "lc": lc_total},
            prog_bar=True,
            on_epoch=True,
        )

    def validation_step(self, batch, batch_idx):
        if batch_idx >= 10:
            return
        frames, flows, bboxs, data_idxs = batch

        # train autoencoder
        z, frames_fake = self._ae_frame(frames)

        # save fig
        frames = frames[:, :, -1]
        frames = frames.permute(0, 2, 3, 1).detach().cpu().numpy()
        frames_fake = frames_fake.permute(0, 2, 3, 1).detach().cpu().numpy()
        frames = ((frames + 1) / 2 * 255).astype(np.uint8)
        frames_fake = ((frames_fake + 1) / 2 * 255).astype(np.uint8)
        cv2.imwrite(f"images/batch{batch_idx}_in.jpg", frames[0])
        cv2.imwrite(f"images/batch{batch_idx}_out.jpg", frames_fake[0])

    # def predict_step(self, batch, batch_idx):
    #     frames, flows, bboxs, data_idxs = batch
    #     batch_size = frames.shape[0]

    #     z, frames_out, flows_out = self._ae(frames, flows, bboxs)
    #     z, s, c = self._cm(z.detach())
    #     preds = []
    #     for i in range(batch_size):
    #         for j in range(self._n_samples_batch):
    #             data = {
    #                 "sample_num": (data_idxs[i] + j).item(),
    #                 "z": z[i, j].cpu().numpy(),
    #                 "c": c[i, j].item(),
    #             }
    #             preds.append(data)

    #     return preds

    def configure_optimizers(self):
        # optimizer
        optim_ae_frame = torch.optim.Adam(
            self._ae_frame.parameters(), self._cfg.optim.lr_rate_ae
        )
        optim_ae_flow = torch.optim.Adam(
            self._ae_flow.parameters(), self._cfg.optim.lr_rate_ae
        )
        optim_cm = torch.optim.Adam(self._cm.parameters(), self._cfg.optim.lr_rate_cm)

        # scheduler
        step_size = (self._cfg.epochs - self._cfg.clustering_start_epoch + 1) // 2
        sch_ae_frame = torch.optim.lr_scheduler.StepLR(
            optim_ae_frame, step_size=step_size, gamma=0.1
        )
        sch_ae_flow = torch.optim.lr_scheduler.StepLR(
            optim_ae_flow, step_size=step_size, gamma=0.1
        )
        sch_cm = torch.optim.lr_scheduler.StepLR(
            optim_cm, step_size=step_size, gamma=0.1
        )
        return (
            [optim_ae_frame, optim_ae_flow, optim_cm],
            [sch_ae_frame, sch_ae_flow, sch_cm],
        )
