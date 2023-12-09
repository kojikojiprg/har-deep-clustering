import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import RoIAlign


class ClusteringModule(nn.Module):
    def __init__(self, cfg, n_samples, n_samples_batch):
        super().__init__()
        self._n_samples = n_samples
        self._n_samples_batch = n_samples_batch

        # get configs
        self._img_w = cfg.img_size.w
        self._img_h = cfg.img_size.h
        i3d_nch = cfg.autoencoder.i3d.nch
        cfg = cfg.clustering
        self._n_clusters = cfg.n_clusters
        self._t_alpha = cfg.alpha
        os = cfg.roialign.output_size

        # layers for visual feature
        self._roi_align = RoIAlign(
            os,
            cfg.roialign.spatial_scale,
            1,
            cfg.roialign.aligned,
        )
        self._emb_visual = nn.Linear(i3d_nch * os * os, cfg.ndf)

        # layer for spatial feature
        self._img_c = nn.Parameter(
            torch.Tensor((self._img_w / 2, self._img_h / 2)), requires_grad=False
        )
        self._emb_spacial = nn.Linear(1, cfg.ndf)

        # centroids
        z = torch.normal(0, 0.1, (cfg.n_clusters, 480 * os * os))
        z = self._emb_visual(z)
        norms = torch.rand((cfg.n_clusters, 1))
        norms = self._emb_spacial(norms)
        self._centroids = nn.ParameterList(
            [
                nn.Parameter(z[i] + norms[i], requires_grad=True)
                for i in range(self._n_clusters)
            ]
        )
        self._target_distribution = torch.zeros((self._n_samples, self._n_clusters))

    @property
    def centroids(self):
        return self._centroids

    @property
    def target_distribution(self):
        return self._target_distribution

    def forward(self, z_vis, bboxs):
        # visual feature
        fy, fx = z_vis.shape[3:5]
        bn, sn = bboxs.shape[0:2]
        bboxs_vis = bboxs.view(-1, 2, 2).clone()
        bboxs_vis *= torch.Tensor((fx / self._img_w, fy / self._img_h)).to(
            next(self.parameters()).device
        )
        bboxs_vis = bboxs_vis.view(bn, -1, 4)
        rois = self._convert_bboxes_to_roi_format(bboxs_vis)

        z_vis = self._roi_align(z_vis[:, :, -1], rois)
        z_vis = z_vis.view(bn * sn, -1)
        z_vis = self._emb_visual(z_vis)
        z_vis = z_vis.view(bn, sn, -1)

        # spacial feature
        z_spc = self._norm_bbox2centor(bboxs)

        # clustering
        s = torch.zeros((bn, sn, self._n_clusters))
        bboxs = bboxs.cpu().numpy()
        for b in range(bn):
            mask_not_nan = ~np.isnan(bboxs[b]).any(axis=1)
            z_spc = self._emb_spacial(z_spc[b][mask_not_nan])
            z = z_vis[b][mask_not_nan] + z_spc
            s[b, : z.shape[0]] = self._student_t(z)

        c = s.argmax(dim=2)

        return z, s, c

    def _convert_bboxes_to_roi_format(self, boxes: torch.Tensor) -> torch.Tensor:
        concat_boxes = torch.cat([b for b in boxes], dim=0)
        temp = []
        for i, b in enumerate(boxes):
            temp.append(torch.full_like(b[:, :1], i))
        ids = torch.cat(temp, dim=0)
        rois = torch.cat([ids, concat_boxes], dim=1)
        rois = rois.nan_to_num(0)
        return rois

    def _student_t(self, z):
        sn = z.shape[0]
        norm = torch.zeros((sn, self._n_clusters), dtype=torch.float32)
        for j in range(self._n_clusters):
            diff = (z - self._centroids[j]).clone()
            norm[:, j] = torch.linalg.vector_norm(diff, dim=1)

        s = torch.zeros((sn, self._n_clusters), dtype=torch.float32)
        s_tmp = torch.zeros_like(s)
        for i in range(sn):
            s_tmp[i] = (1 + norm[i] / self._t_alpha) ** -((self._t_alpha + 1) / 2)
        s_tmp_sum = s_tmp.sum(dim=1)
        s = s_tmp / s_tmp_sum.view(-1, 1)

        return s

    def update_target_distribution(self, s, batch_idxs):
        sn = s.shape[1]

        for b, batch_idx in enumerate(batch_idxs):
            s_sums = s[b].sum(dim=0)
            for i in range(sn):
                targets = torch.zeros(self._n_clusters)
                for j in range(self._n_clusters):
                    sij = s[b, i, j]
                    targets[j] = sij**2 / s_sums[i]
                t_sums = targets.sum(dim=0)
                targets = targets / t_sums

                ti = batch_idx * self._n_samples_batch + i  # target idx
                self._target_distribution[ti] = targets

    def _norm_bbox2centor(self, bboxs):
        self._img_c = self._img_c.to(next(self.parameters()).device)
        bn, sn = bboxs.shape[:2]

        bboxs = bboxs.view(bn, sn, 2, 2)
        centors = bboxs[:, :, 0] + (bboxs[:, :, 1] - bboxs[:, :, 0]) / 2
        norms = torch.zeros((bn, sn, 1)).to(next(self.parameters()).device)
        for b in range(bn):
            diffs = (centors[b] - self._img_c).clone()
            norms_batch = torch.linalg.vector_norm(diffs, dim=1)
            norms_batch = norms_batch / torch.linalg.vector_norm(self._img_c)
            norms[b] = norms_batch.view(bn, sn, 1)

        return norms.detach()
