import torch.nn as nn

from .i3d import InceptionI3d


class Encoder(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        cfg = cfg.autoencoder

        self._i3d = InceptionI3d(
            in_channels=in_channels, final_endpoint=cfg.i3d.final_endpoint
        )

    def forward(self, imgs):
        return self._i3d(imgs)


class Decoder(nn.Module):
    def __init__(self, cfg, out_channels):
        super().__init__()
        self._out_w = cfg.img_size.w
        self._out_h = cfg.img_size.h
        ndf = cfg.autoencoder.ndf
        self._out_channels = out_channels

        self.net = nn.Sequential(
            # TODO: setup network automatically
            nn.ConvTranspose2d(480, ndf * 8, 4, 2, 1, bias=False),
            # nn.ConvTranspose2d(480, ndf * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.1, True),
            # (ndf*8) x h*2 x w*2
            nn.ConvTranspose2d(ndf * 8, ndf * 4, 4, 2, 1, bias=False),
            # nn.ConvTranspose2d(ndf * 8, ndf * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.1, True),
            # (ndf*4) x h*4 x w*4
            nn.ConvTranspose2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, True),
            # (ndf*2) x h*8 x w*8
            nn.ConvTranspose2d(ndf * 2, ndf * 1, 1, 1, 0, bias=False),
            # nn.ConvTranspose2d(ndf * 2, ndf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 1),
            nn.LeakyReLU(0.1, True),
            # (ndf*1) x h*8 x w*8
            nn.ConvTranspose2d(ndf * 1, out_channels, 1, 1, 0, bias=False),
            # nn.ConvTranspose2d(ndf * 1, out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # (out_chennels) x h*8 x w*8``
        )

    def forward(self, z):
        return self.net(z)


class Autoencoder(nn.Module):
    def __init__(self, cfg, n_channels):
        super().__init__()
        self._e = Encoder(cfg, n_channels)
        self._d = Decoder(cfg, n_channels)

    @property
    def E(self):
        return self._e

    @property
    def D(self):
        return self._d

    def forward(self, imgs):
        z = self._e(imgs)
        fake_imgs = self._d(z[:, :, -1])
        return z, fake_imgs
