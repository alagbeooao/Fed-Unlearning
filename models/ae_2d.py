"""
2D convolutional autoencoder for unsupervised anomaly detection in brain MRI.

Baur et al. 2021, Medical Image Analysis 69:101952.

This is Baur's spatial variant, not the dense one. His dense AE has a
fully-connected low-dimensional bottleneck; here the bottleneck stays
convolutional, 256x8x8 for a 128x128 input, with no dense layer anywhere.

Four stride-2 conv blocks down (1-32-64-128-256), mirrored on the way back up,
L1 reconstruction loss. At inference the per-pixel error |x - recon| is the
anomaly score.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> LeakyReLU, with optional stride-2 downsampling."""

    def __init__(self, in_ch, out_ch, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class UpBlock(nn.Module):
    """Transposed-conv upsample -> BatchNorm -> LeakyReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                                     stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.up(x)))


class AE2D(nn.Module):
    """2D convolutional (spatial) autoencoder.

    Args:
        in_channels: input channels (1 for FLAIR-only).
        base_channels: channels in the first encoder block.
        latent_channels: channels at the bottleneck.

    For 128x128 input the bottleneck is (B, latent_channels, 8, 8).
    """

    def __init__(self, in_channels=1, base_channels=32, latent_channels=256):
        super().__init__()
        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 4   # 128
        c4 = latent_channels     # 256

        # Encoder: 128 -> 64 -> 32 -> 16 -> 8
        self.enc1 = ConvBlock(in_channels, c1, downsample=True)
        self.enc2 = ConvBlock(c1, c2, downsample=True)
        self.enc3 = ConvBlock(c2, c3, downsample=True)
        self.enc4 = ConvBlock(c3, c4, downsample=True)

        # Decoder: 8 -> 16 -> 32 -> 64 -> 128
        self.dec1 = UpBlock(c4, c3)
        self.dec2 = UpBlock(c3, c2)
        self.dec3 = UpBlock(c2, c1)
        self.dec4 = nn.ConvTranspose2d(c1, in_channels, kernel_size=4,
                                       stride=2, padding=1)

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        return x

    def decode(self, z):
        z = self.dec1(z)
        z = self.dec2(z)
        z = self.dec3(z)
        z = self.dec4(z)
        return z

    def forward(self, x):
        return self.decode(self.encode(x))


def reconstruction_loss(pred, target, mode="l1"):
    """Scalar reconstruction loss.

    Args:
        pred, target: (B, C, H, W).
        mode: 'l1' or 'l2'.
    """
    if mode == "l1":
        return F.l1_loss(pred, target, reduction="mean")
    if mode == "l2":
        return F.mse_loss(pred, target, reduction="mean")
    raise ValueError(f"Unknown loss mode: {mode}")


def reconstruction_error_map(pred, target, mode="l1"):
    """Per-pixel reconstruction error (channel-averaged), not reduced.

    Used at inference for anomaly localisation.

    Returns:
        (B, H, W) error map.
    """
    if mode == "l1":
        err = (pred - target).abs()
    elif mode == "l2":
        err = (pred - target) ** 2
    else:
        raise ValueError(f"Unknown loss mode: {mode}")
    return err.mean(dim=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
