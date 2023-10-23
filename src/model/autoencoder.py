import torch
import torch.nn as nn
from torchvision.ops import RoIAlign

from .i3d import InceptionI3d


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg = cfg.autoencoder

        self._i3d_frame = InceptionI3d(
            in_channels=3, final_endpoint=cfg.i3d.final_endpoint
        )
        self._i3d_flow = InceptionI3d(
            in_channels=2, final_endpoint=cfg.i3d.final_endpoint
        )
        self._roi_align = RoIAlign(
            cfg.roialign.output_size,
            cfg.roialign.spatial_scale,
            1,
            cfg.roialign.aligned,
        )

    def forward(self, frames, flows, bboxs):
        # forward i3d
        frames = self._i3d_frame(frames)
        flows = self._i3d_flow(flows)
        f = frames + flows

        # format bbox
        h, w = frames.shape[3:5]
        fy, fx = f.shape[3:5]
        b = bboxs.shape[0]
        bboxs = bboxs.view(-1, 2, 2)
        bboxs *= torch.Tensor((fx / w, fy / h)).to(next(self.parameters()).device)
        bboxs = bboxs.view(b, -1, 4)
        bboxs = self._convert_bboxes_to_roi_format(bboxs)

        # roi align
        f = f[:, :, -1]
        return self._roi_align(f, bboxs)

    def _convert_bboxes_to_roi_format(self, boxes: torch.Tensor) -> torch.Tensor:
        concat_boxes = torch.cat([b for b in boxes], dim=0)
        temp = []
        for i, b in enumerate(boxes):
            temp.append(torch.full_like(b[:, :1], i))
        ids = torch.cat(temp, dim=0)
        rois = torch.cat([ids, concat_boxes], dim=1)
        rois = rois.nan_to_num(0)
        return rois


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._out_w = cfg.fake_size.w
        self._out_h = cfg.fake_size.h
        ndf = cfg.autoencoder.ndf
        self._out_channels = 5  # frame + flow

        self.net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(480, ndf * 8, 4, 3, (0, 2), bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.1, True),
            # state size. ``(ngf*8) x 16 x 12``
            nn.ConvTranspose2d(ndf * 8, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.1, True),
            # state size. ``(ngf*4) x 32 x 24``
            nn.ConvTranspose2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, True),
            # state size. ``(ngf*2) x 64 x 48``
            nn.ConvTranspose2d(ndf * 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1, True),
            # state size. ``(ngf) x 128 x 96``
            nn.ConvTranspose2d(ndf, self._out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. ``(nc) x 256 x 192``
        )

    def forward(self, z):
        n = z.shape[0]
        out = self.net(z)
        out = out.view(n, self._out_channels, self._out_h, self._out_w)
        # frames = out[:, :3]
        # flows = out[:, 3:]
        return out[:, :3], out[:, 3:]


class Autoencoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._encoder = Encoder(cfg)
        self._decoder = Decoder(cfg)

    def forward(self, frames, flows, bboxs):
        z = self._encoder(frames, flows, bboxs)
        frames_d, flows_d = self._decoder(z)

        # adjust shapes
        b, n = bboxs.shape[:2]
        c, sy, sx = z.shape[1:]
        z = z.view(b, n, c, sy, sx)
        frames_d = frames_d.view(b, n, 3, 256, 192)
        flows_d = flows_d.view(b, n, 2, 256, 192)

        return z, frames_d, flows_d
