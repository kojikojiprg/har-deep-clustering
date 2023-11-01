import torch
import torch.nn as nn
from torchvision.ops import RoIAlign


class ClusteringModule(nn.Module):
    def __init__(self, cfg, n_samples, n_samples_batch):
        super().__init__()
        self._n_samples = n_samples
        self._n_samples_batch = n_samples_batch

        self._img_w = cfg.img_size.w
        self._img_h = cfg.img_size.h

        cfg = cfg.clustering
        self._n_clusters = cfg.n_clusters
        self._t_alpha = cfg.alpha

        self._centroids = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn((cfg.ndf2), dtype=torch.float32), requires_grad=True
                )
                for _ in range(self._n_clusters)
            ]
        )
        self._target_distribution = torch.zeros((self._n_samples, self._n_clusters))

        os = cfg.roialign.output_size
        self._emb = nn.Sequential(
            nn.Linear(480 * os * os, cfg.ndf1),
            nn.Linear(cfg.ndf1, cfg.ndf2),
        )
        self._roi_align = RoIAlign(
            os,
            cfg.roialign.spatial_scale,
            1,
            cfg.roialign.aligned,
        )

    @property
    def centroids(self):
        return self._centroids

    @property
    def target_distribution(self):
        return self._target_distribution

    def forward(self, z, bboxs):
        # format bbox
        fy, fx = z.shape[3:5]
        b, sn = bboxs.shape[0:2]
        bboxs = bboxs.view(-1, 2, 2)
        bboxs *= torch.Tensor((fx / self._img_w, fy / self._img_h)).to(
            next(self.parameters()).device
        )
        bboxs = bboxs.view(b, -1, 4)
        bboxs = self._convert_bboxes_to_roi_format(bboxs)

        # roi align
        z = self._roi_align(z[:, :, -1], bboxs)

        # clustering
        z = z.view(b * sn, -1)
        z = self._emb(z)
        z = z.view(b, sn, -1)

        s = self._student_t(z)
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
        bn, sn = z.shape[:2]
        norm = torch.zeros((bn, sn, self._n_clusters), dtype=torch.float32)
        norm_tmp = torch.zeros_like(norm)
        for b in range(bn):
            for j in range(self._n_clusters):
                diff = (z[b] - self._centroids[j]).clone()
                norm_tmp[b, :, j] = torch.linalg.vector_norm(diff, dim=1)
        norm = norm_tmp

        s = torch.zeros((bn, sn, self._n_clusters), dtype=torch.float32)
        s_tmp = torch.zeros_like(s)
        for b in range(bn):
            for i in range(sn):
                s_tmp[b, i] = ((1 + norm[b, i]) / self._t_alpha) ** -(
                    (self._t_alpha + 1) / 2
                )
            s_tmp_t = s_tmp[b].T.detach()
            s_tmp_sum = s_tmp[b].sum(dim=1).detach()
            s_tmp[b] = (s_tmp_t / s_tmp_sum).T
        s = s_tmp

        return s

    def update_target_distribution(self, s, batch_idxs):
        sample_nums = s.shape[1]
        s = s.view(-1, self._n_clusters).detach()
        s_sums = s.sum(dim=0)  # Sigma_i s_ij (n_clusters,)

        targets = torch.zeros((self._n_samples, self._n_clusters))
        for i, batch_idx in enumerate(batch_idxs):
            for sn in range(sample_nums):
                ti = batch_idx * self._n_samples_batch + sn  # target idx
                si = i * self._n_clusters + sn  # soft idx
                for j in range(self._n_clusters):
                    sij = s[si, j]
                    targets[ti, j] = sij**2 / s_sums[j]
                targets[ti] = targets[ti] / targets[ti].sum(dim=0)
                self._target_distribution[ti] = targets[ti]
