import torch
import torch.nn as nn


class ClusteringModule(nn.Module):
    def __init__(self, cfg, n_samples, n_samples_batch):
        super().__init__()
        cfg = cfg.clustering
        self._n_clusters = cfg.n_clusters
        self._n_samples = n_samples
        self._n_samples_batch = n_samples_batch
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

        self._emb = nn.Sequential(
            nn.Linear(480 * 5 * 5, cfg.ndf1),
            nn.Linear(cfg.ndf1, cfg.ndf2),
        )

    @property
    def centroids(self):
        return self._centroids

    @property
    def target_distribution(self):
        return self._target_distribution

    def forward(self, z):
        b, sn = z.shape[:2]
        z = z.view(b * sn, -1)
        z = self._emb(z)
        z = z.view(b, sn, -1)

        s = self._student_t(z)
        c = s.argmax(dim=2)
        return z, s, c

    def _student_t(self, z):
        b, sn = z.shape[:2]
        norm = torch.zeros((b, sn, self._n_clusters), dtype=torch.float32)
        norm_tmp = torch.zeros_like(norm)
        for i in range(b):
            for j in range(self._n_clusters):
                diff = (z[i] - self._centroids[j]).clone()
                norm_tmp[i, :, j] = torch.linalg.vector_norm(diff, dim=1)
        norm = norm_tmp

        s = torch.zeros((b, sn, self._n_clusters), dtype=torch.float32)
        s_tmp = torch.zeros_like(s)
        for i in range(b):
            for j in range(sn):
                s_tmp[i, j] = ((1 + norm[i, j]) / self._t_alpha) ** -(
                    (self._t_alpha + 1) / 2
                )
            s_tmp_t = s_tmp[i].T.detach()
            s_tmp_sum = s_tmp[i].sum(dim=1).detach()
            s_tmp[i] = (s_tmp_t / s_tmp_sum).T
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
