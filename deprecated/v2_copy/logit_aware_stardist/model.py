from __future__ import annotations
import torch
from torch import nn


def conv_block(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None):
    """Conv block."""
    if p is None:
        p = k // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class LogitAwareStarDist(nn.Module):
    """
    Simple UNet-like backbone with prob and ray heads.
    Input channels: image (3) + prior/logit (1).
    """

    def __init__(self, in_ch: int = 4, base: int = 32, n_rays: int = 32):
        """Initialize the instance."""
        super().__init__()
        self.n_rays = n_rays

        self.enc1 = nn.Sequential(conv_block(in_ch, base), conv_block(base, base))
        self.enc2 = nn.Sequential(conv_block(base, base * 2, s=2), conv_block(base * 2, base * 2))
        self.enc3 = nn.Sequential(conv_block(base * 2, base * 4, s=2), conv_block(base * 4, base * 4))

        self.bottleneck = nn.Sequential(conv_block(base * 4, base * 8, s=2), conv_block(base * 8, base * 8))

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(conv_block(base * 8, base * 4), conv_block(base * 4, base * 4))
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(conv_block(base * 4, base * 2), conv_block(base * 2, base * 2))
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(conv_block(base * 2, base), conv_block(base, base))

        self.prob_head = nn.Conv2d(base, 1, kernel_size=1)
        self.ray_head = nn.Conv2d(base, n_rays, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """Forward."""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        prob = self.prob_head(d1)
        rays = self.ray_head(d1)
        return prob, rays
