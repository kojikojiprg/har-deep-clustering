import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from .i3d import InceptionI3d

sys.path.append("src")
from utils.video import flow_to_rgb


class Encoder(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        final_endpoint = cfg.autoencoder.i3d.final_endpoint
        self._i3d = InceptionI3d(in_channels=in_channels, final_endpoint=final_endpoint)

    def forward(self, imgs):
        z = self._i3d(imgs)

        return z


class Decoder(nn.Module):
    def __init__(self, cfg, out_channels):
        super().__init__()
        self._out_w = cfg.img_size.w
        self._out_h = cfg.img_size.h
        i3d_nch = cfg.autoencoder.i3d.nch
        nch = cfg.autoencoder.nch
        self._out_channels = out_channels

        krnl = (1, 4, 4)
        strd = (1, 2, 2)
        pad = (0, 1, 1)
        self.net = nn.Sequential(
            # TODO: setup network automatically
            nn.ConvTranspose3d(i3d_nch, nch * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(nch * 8),
            nn.LeakyReLU(0.1, True),
            # (ndf*8) x s*2 x h*2 x w*2
            nn.ConvTranspose3d(nch * 8, nch * 4, krnl, strd, pad, bias=False),
            nn.BatchNorm3d(nch * 4),
            nn.LeakyReLU(0.1, True),
            # (ndf*4) x s*2 x h*4 x w*4
            nn.ConvTranspose3d(nch * 4, nch * 2, krnl, strd, pad, bias=False),
            nn.BatchNorm3d(nch * 2),
            nn.LeakyReLU(0.1, True),
            # (ndf*2) x s*2 x h*8 x w*8
            nn.ConvTranspose3d(nch * 2, nch * 1, 1, 1, 0, bias=False),
            nn.BatchNorm3d(nch * 1),
            nn.LeakyReLU(0.1, True),
            # (ndf*1) x s*2 x h*8 x w*8
            nn.ConvTranspose3d(nch * 1, out_channels, 1, 1, 0, bias=False),
            nn.Tanh(),
            # (out_chennels) x s*2 x h*8 x w*8``
        )

    def forward(self, z):
        imgs = self.net(z)
        return imgs


class Autoencoder(nn.Module):
    def __init__(self, cfg, n_channels):
        super().__init__()
        self._seq_len = cfg.seq_len
        self._e = Encoder(cfg, n_channels)
        self._d = Decoder(cfg, n_channels)

    @property
    def E(self):
        return self._e

    @property
    def D(self):
        return self._d

    def forward(self, imgs):
        # imgs(b, c, seq_len, h, w)

        z = self._e(imgs)
        # z(b, 480, 10, fy, fx) from Mixed_3c

        fake_imgs = self._d(z)
        # fake_imgs(b, c, seq_len, h, w)
        return z, fake_imgs


class AutoencoderModule(LightningModule):
    def __init__(self, cfg: SimpleNamespace, datatype: str, checkpoint_dir: str):
        super().__init__()
        self._cfg = cfg

        if datatype == "frame":
            n_channels = 3
        elif datatype == "flow":
            n_channels = 2
        else:
            raise ValueError
        self._datatype = datatype

        self._ae = Autoencoder(cfg, n_channels)
        self._lr = nn.MSELoss()

        self._callbacks = [
            ModelCheckpoint(
                checkpoint_dir,
                filename=f"ae_{datatype}_seq{cfg.seq_len}_lc_min",
                monitor="lr",
                mode="min",
                save_last=True,
            ),
        ]
        last_name = f"ae_{datatype}_seq{cfg.seq_len}_last"
        self._callbacks[0].CHECKPOINT_NAME_LAST = last_name

    @property
    def callbacks(self) -> list:
        return self._callbacks

    def training_step(self, batch, batch_idx):
        frames, flows, bboxs, norms, data_idxs = batch
        if self._datatype == "frame":
            x = frames
        elif self._datatype == "flow":
            x = flows

        _, fakes = self._ae(x)
        lr = self._lr(x, fakes)
        self.log("lr", lr, on_epoch=True)
        return lr

    def validation_step(self, batch, batch_idx):
        frames, flows, bboxs, norms, data_idxs = batch
        if self._datatype == "frame":
            x = frames
        elif self._datatype == "flow":
            x = flows

        # forward autoencoder
        _, fakes = self._ae(x)

        # save fig
        if self._datatype == "frame":
            frames = frames[:, :, -1]
            frames = frames.permute(0, 2, 3, 1).detach().cpu().numpy()
            fakes = fakes[:, :, -1]
            fakes = fakes.permute(0, 2, 3, 1).detach().cpu().numpy()
            frames = ((frames + 1) / 2 * 255).astype(np.uint8)
            fakes = ((fakes + 1) / 2 * 255).astype(np.uint8)
            cv2.imwrite(f"images/batch{batch_idx}_framein.jpg", frames[0])
            cv2.imwrite(f"images/batch{batch_idx}_frameout.jpg", fakes[0])

        elif self._datatype == "flow":
            flow = flows[:, :, -1].permute(0, 2, 3, 1).cpu().numpy()[0]
            flow = flow_to_rgb(flow)
            fakes = fakes[:, :, -1]
            flow_fake = fakes.permute(0, 2, 3, 1).cpu().numpy()[0]
            flow_fake = flow_to_rgb(flow_fake)
            cv2.imwrite(f"images/batch{batch_idx}_flowin.jpg", flow)
            cv2.imwrite(f"images/batch{batch_idx}_flowout.jpg", flow_fake)

    def configure_optimizers(self):
        # optimizer
        optim = Adam(self._ae.parameters(), self._cfg.optim.lr)
        # scheduler
        step_size = self._cfg.epochs // 2
        sch = StepLR(optim, step_size, 0.1)

        return [optim], [sch]
