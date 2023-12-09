import sys
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from numpy.typing import NDArray
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

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
        self._update_interval = cfg.update_interval
        self._n_samples = n_samples
        self._n_samples_batch = n_samples_batch
        self._lmd_cm = cfg.optim.lmd_cm

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

    def forward(self, flows, bboxs):
        z, fake_flows = self._ae_flow(flows)
        z, s, c = self._cm(z.detach(), bboxs)

        return fake_flows, z, s, c

    def _calc_lc(self, s, bboxs, data_idxs):
        # calc clustering loss
        lc_total = 0
        # if self.current_epoch + 1 >= self._cfg.clustering_start_epoch:
        for b, data_idx in enumerate(data_idxs):
            ti = data_idx * self._n_samples_batch
            tmp_t = self._cm.target_distribution[ti : ti + self._n_samples_batch]
            tmp_s = s[b]

            bboxs_batch = bboxs[b].cpu().numpy()
            mask_not_nan = ~np.isnan(bboxs_batch).any(axis=1)
            tmp_t = tmp_t[mask_not_nan]
            tmp_s = tmp_s[mask_not_nan]
            lc_total = lc_total + self._lc(tmp_s.log(), tmp_t.detach())

        return lc_total

    def training_step(self, batch, batch_idx, optimizer_idx):
        _, flows, bboxs, data_idxs = batch

        if optimizer_idx == 0:  # clustering
            self._cm.train()
            fake_flows, _, s, _ = self(flows, bboxs)

            if self.current_epoch % self._update_interval == 0:  # update target
                self._cm.update_target_distribution(s, data_idxs)

            # if self.current_epoch + 1 < self._cfg.clustering_start_epoch:
            #     return None  # skip training clustering module
            lc_total = self._calc_lc(s, bboxs, data_idxs)
            self.log("lc", lc_total, prog_bar=True, on_epoch=True)
            return lc_total

        elif optimizer_idx == 1:  # flow decoder
            _, fake_flows = self._ae_flow(flows)
            lr_flow = self._lr(flows, fake_flows)
            self.log("lr_flow", lr_flow, on_epoch=True)
            return lr_flow

        elif optimizer_idx == 2:  # flow encoder
            self._cm.eval()
            fake_flows, _, s, _ = self(flows, bboxs)
            lr_flow = self._lr(flows, fake_flows)
            lc_total = self._calc_lc(s, bboxs, data_idxs)
            return lr_flow + lc_total * self._lmd_cm

    def validation_step(self, batch, batch_idx):
        frames, flows, bboxs, data_idxs = batch

        # forward autoencoder
        _, flows_fake = self._ae_flow(flows)

        # save fig
        flow = flows[:, :, -1].permute(0, 2, 3, 1).cpu().numpy()[0]
        flow = flow_to_rgb(flow)
        flows_fake = flows_fake[:, :, -1]
        flow_fake = flows_fake.permute(0, 2, 3, 1).cpu().numpy()[0]
        flow_fake = flow_to_rgb(flow_fake)
        cv2.imwrite(f"images/batch{batch_idx}_flowin.jpg", flow)
        cv2.imwrite(f"images/batch{batch_idx}_flowout.jpg", flow_fake)

    def predict_step(self, batch, batch_idx):
        frames, flows, bboxs, data_idxs = batch
        batch_size = flows.shape[0]

        _, z, _, c = self(flows, bboxs.clone())
        preds = []
        for i in range(batch_size):
            for j in range(self._n_samples_batch):
                bbox = bboxs[i, j].cpu().numpy()
                if np.any(np.isnan(bbox)):
                    continue
                data = {
                    "sample_num": (data_idxs[i] + j).item(),
                    "z": z[i, j].cpu().numpy(),
                    "bbox": bbox,
                    "c": c[i, j].item(),
                }
                preds.append(data)

        return preds

    def configure_optimizers(self):
        # optimizer
        optim_cm = Adam(self._cm.parameters(), self._cfg.optim.lr_rate_cm)
        optim_d_flow = Adam(
            self._ae_flow.D.parameters(), self._cfg.optim.lr_rate_ae_flow
        )
        optim_e_flow = Adam(
            self._ae_flow.E.parameters(), self._cfg.optim.lr_rate_ae_flow
        )

        # scheduler
        step_size = self._cfg.epochs // 2
        sch_cm = StepLR(optim_cm, step_size, 0.1)
        sch_d_flow = StepLR(optim_d_flow, step_size, 0.1)
        sch_e_flow = StepLR(optim_e_flow, step_size, 0.1)

        return (
            [optim_cm, optim_d_flow, optim_e_flow],
            [sch_cm, sch_d_flow, sch_e_flow],
        )
