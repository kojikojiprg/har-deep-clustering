import sys
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

sys.path.append("src")
from utils.video import flow_to_rgb


class DeepClusteringModel(LightningModule):
    def __init__(
        self,
        cfg: SimpleNamespace,
        n_samples: int,
        n_samples_batch: int,
        checkpoint_dir: Optional[str] = None,
    ):
        super().__init__()
        self._cfg = cfg
        self._n_samples = n_samples
        self._n_samples_batch = n_samples_batch
        self._lmd_cm = cfg.optim.lmd_cm

        self._ae_frame = Autoencoder(cfg, 3)
        self._ae_flow = Autoencoder(cfg, 2)
        self._cm = ClusteringModule(cfg, n_samples, n_samples_batch)

        self._lr = nn.MSELoss()
        self._lc = nn.KLDivLoss(reduction="sum")

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
    def target_distribution(self) -> NDArray:
        return self._cm.target_distribution.detach().cpu().numpy()

    @property
    def centroids(self) -> NDArray:
        return np.array([c.detach().numpy() for c in iter(self._cm.centroids.cpu())])

    def forward(self, frames, flows, bboxs):
        z_frame, fake_frames = self._ae_frame(frames)
        z_flow, fake_flows = self._ae_flow(flows)
        z = z_frame.detach() + z_flow.detach()
        z, s, c = self._cm(z, bboxs)

        return fake_frames, fake_flows, z, s, c

    def training_step(self, batch, batch_idx, optimizer_idx):
        frames, flows, bboxs, data_idxs = batch

        # forward model
        fake_frames, fake_flows, _, s, _ = self(frames, flows, bboxs)

        if optimizer_idx == 0:  # update target
            # if self.current_epoch + 1 >= self._cfg.clustering_start_epoch:
            self._cm.update_target_distribution(s, data_idxs)

        # calc autrocoder loss
        if optimizer_idx in [0, 2]:
            lr_frame = self._lr(frames, fake_frames)
        if optimizer_idx in [1, 3]:
            lr_flow = self._lr(flows, fake_flows)

        # calc clustering loss
        if optimizer_idx in [0, 1, 4]:
            lc_total = 0
            # if self.current_epoch + 1 >= self._cfg.clustering_start_epoch:
            for i, data_idx in enumerate(data_idxs):
                idx = data_idx * self._n_samples_batch
                tmp_target = self._cm.target_distribution[
                    idx : idx + self._n_samples_batch
                ]
                tmp_s = s[i]
                lc_total = lc_total + self._lc(tmp_s.log(), tmp_target)

        if optimizer_idx == 0:  # frame encoder
            return lr_frame + lc_total * self._lmd_cm
        elif optimizer_idx == 1:  # flow encoder
            return lr_flow + lc_total * self._lmd_cm
        elif optimizer_idx == 2:  # frame decoder
            self.log("lr_frame", lr_frame, on_epoch=True)
            return lr_frame
        elif optimizer_idx == 3:  # flow decoder
            self.log("lr_flow", lr_flow, on_epoch=True)
            return lr_flow
        elif optimizer_idx == 4:  # clustering
            # if self.current_epoch + 1 < self._cfg.clustering_start_epoch:
            #     return None  # skip training clustering module
            self.log("lc", lc_total, prog_bar=True, on_epoch=True)
            return lc_total
        else:
            raise ValueError(f"optimizer_idx {optimizer_idx}")

    def validation_step(self, batch, batch_idx):
        frames, flows, bboxs, data_idxs = batch

        # forward autoencoder
        _, frames_fake = self._ae_frame(frames)
        _, flows_fake = self._ae_flow(flows)

        # save fig
        frames = frames[:, :, -1]
        frames = frames.permute(0, 2, 3, 1).detach().cpu().numpy()
        frames_fake = frames_fake[:, :, -1]
        frames_fake = frames_fake.permute(0, 2, 3, 1).detach().cpu().numpy()
        frames = ((frames + 1) / 2 * 255).astype(np.uint8)
        frames_fake = ((frames_fake + 1) / 2 * 255).astype(np.uint8)
        cv2.imwrite(f"images/batch{batch_idx}_framein.jpg", frames[0])
        cv2.imwrite(f"images/batch{batch_idx}_frameout.jpg", frames_fake[0])

        flow = flows[:, :, -1].permute(0, 2, 3, 1).cpu().numpy()[0]
        flow = flow_to_rgb(flow)
        flows_fake = flows_fake[:, :, -1]
        flow_fake = flows_fake.permute(0, 2, 3, 1).cpu().numpy()[0]
        flow_fake = flow_to_rgb(flow_fake)
        cv2.imwrite(f"images/batch{batch_idx}_flowin.jpg", flow)
        cv2.imwrite(f"images/batch{batch_idx}_flowout.jpg", flow_fake)

    def predict_step(self, batch, batch_idx):
        frames, flows, bboxs, data_idxs = batch
        batch_size = frames.shape[0]

        z_frame, frames_out = self._ae_frame(frames)
        z_flow, flows_out = self._ae_flow(flows)
        z = z_frame.detach() + z_flow.detach()
        z, s, c = self._cm(z, bboxs)
        preds = []
        for i in range(batch_size):
            for j in range(self._n_samples_batch):
                data = {
                    "sample_num": (data_idxs[i] + j).item(),
                    "z": z[i, j].cpu().numpy(),
                    "c": c[i, j].item(),
                }
                preds.append(data)

        return preds

    def configure_optimizers(self):
        # optimizer
        optim_e_frame = torch.optim.Adam(
            self._ae_frame.E.parameters(), self._cfg.optim.lr_rate_ae_frame
        )
        optim_e_flow = torch.optim.Adam(
            self._ae_flow.E.parameters(), self._cfg.optim.lr_rate_ae_flow
        )
        optim_d_frame = torch.optim.Adam(
            self._ae_frame.D.parameters(), self._cfg.optim.lr_rate_ae_frame
        )
        optim_d_flow = torch.optim.Adam(
            self._ae_flow.D.parameters(), self._cfg.optim.lr_rate_ae_flow
        )
        optim_cm = torch.optim.Adam(self._cm.parameters(), self._cfg.optim.lr_rate_cm)

        # scheduler
        step_size = self._cfg.epochs // 2
        sch_e_frame = torch.optim.lr_scheduler.StepLR(
            optim_e_frame, step_size=step_size, gamma=0.1
        )
        sch_e_flow = torch.optim.lr_scheduler.StepLR(
            optim_e_flow, step_size=step_size, gamma=0.1
        )
        sch_d_frame = torch.optim.lr_scheduler.StepLR(
            optim_d_frame, step_size=step_size, gamma=0.1
        )
        sch_d_flow = torch.optim.lr_scheduler.StepLR(
            optim_d_flow, step_size=step_size, gamma=0.1
        )
        sch_cm = torch.optim.lr_scheduler.StepLR(
            optim_cm, step_size=step_size, gamma=0.1
        )

        return (
            [optim_e_frame, optim_e_flow, optim_d_frame, optim_d_flow, optim_cm],
            [sch_e_frame, sch_e_flow, sch_d_frame, sch_d_flow, sch_cm],
        )
