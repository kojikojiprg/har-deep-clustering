import os
from types import SimpleNamespace
from typing import Optional

import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from .abstract_deep_clustering import AbstractDeepClusteringModel
from .autoencoder import AutoencoderModule


class DeepClusteringModel(AbstractDeepClusteringModel):
    def __init__(
        self,
        model_type: str,
        cfg: SimpleNamespace,
        n_samples: int,
        n_samples_batch: int,
        checkpoint_dir: Optional[str] = None,
        version: Optional[int] = None,
        load_autoencoder_checkpoint: bool = True,
    ):
        super().__init__(
            model_type, cfg, n_samples, n_samples_batch, checkpoint_dir, version
        )
        if load_autoencoder_checkpoint:
            ae_frame_ckpt = os.path.join(
                checkpoint_dir, "autoencoder", f"ae_frame_seq{cfg.seq_len}_last.ckpt"
            )
            self._ae_frame = AutoencoderModule.load_from_checkpoint(
                ae_frame_ckpt, cfg=cfg, datatype="frame"
            )
            ae_flow_ckpt = os.path.join(
                checkpoint_dir, "autoencoder", f"ae_flow_seq{cfg.seq_len}_last.ckpt"
            )
            self._ae_flow = AutoencoderModule.load_from_checkpoint(
                ae_flow_ckpt, cfg=cfg, datatype="flow"
            )
        else:
            self._ae_frame = AutoencoderModule(cfg, "frame")
            self._ae_flow = AutoencoderModule(cfg, "flow")

    def forward(self, flows, bboxs, norms):
        z, fake_flows = self._ae_flow(flows)
        z, s, c = self.cm(z.detach(), bboxs, norms)

        return fake_flows, z, s, c

    def training_step(self, batch, batch_idx, optimizer_idx):
        _, flows, bboxs, norms, data_idxs = batch

        if optimizer_idx == 0:  # clustering
            fake_flows, _, s, _ = self(flows, bboxs, norms)

            if self.current_epoch % self.update_interval == 0:  # update target
                self.cm.update_target_distribution(s, data_idxs)

            # if self.current_epoch + 1 < self._cfg.clustering_start_epoch:
            #     return None  # skip training clustering module
            lc_total = self.calc_lc(s, bboxs, data_idxs)
            self.log("lc", lc_total, prog_bar=True, on_epoch=True)
            return lc_total

        elif optimizer_idx == 1:  # flow encoder
            self.cm.requires_grad_(False)
            fake_flows, _, s, _ = self(flows, bboxs, norms)
            lr_flow = self.lr(flows, fake_flows)
            lc_total = self.calc_lc(s, bboxs, data_idxs)
            self.cm.requires_grad_(True)
            return lr_flow + lc_total * self.lmd_cm

        elif optimizer_idx == 2:  # flow decoder
            _, fake_flows = self._ae_flow(flows)
            lr_flow = self.lr(flows, fake_flows)
            self.log("lr_flow", lr_flow, on_epoch=True)
            return lr_flow

    # def validation_step(self, batch, batch_idx):
    #     frames, flows, bboxs, norms, data_idxs = batch

    def predict_step(self, batch, batch_idx):
        frames, flows, bboxs, norms, data_idxs = batch
        batch_size = flows.shape[0]

        _, z, _, c = self(flows, bboxs.clone(), norms)
        preds = []
        for i in range(batch_size):
            for j in range(self.n_samples_batch):
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
        optim_cm = Adam(self.cm.parameters(), self.cfg.optim.lr_rate_cm)
        optim_e_flow = Adam(
            self._ae_flow.E.parameters(), self.cfg.optim.lr_rate_ae_flow
        )
        optim_d_flow = Adam(
            self._ae_flow.D.parameters(), self.cfg.optim.lr_rate_ae_flow
        )

        # scheduler
        step_size = self.cfg.epochs // 2
        sch_cm = StepLR(optim_cm, step_size, 0.1)
        sch_e_flow = StepLR(optim_e_flow, step_size, 0.1)
        sch_d_flow = StepLR(optim_d_flow, step_size, 0.1)

        return (
            [optim_cm, optim_e_flow, optim_d_flow],
            [sch_cm, sch_e_flow, sch_d_flow],
        )
