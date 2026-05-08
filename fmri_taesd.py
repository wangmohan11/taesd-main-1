#!/usr/bin/env python3
"""
Tiny 3D autoencoder for fMRI volumes.

This is a standalone 3D counterpart to TAESD-style encoders. It is meant for
latent diffusion over fMRI volumes shaped [B, 16, 80, 96, 80].
"""
import torch
import torch.nn as nn


def conv3d(n_in, n_out, **kwargs):
    return nn.Conv3d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block3D(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv3d(n_in, n_out),
            nn.ReLU(),
            conv3d(n_out, n_out),
            nn.ReLU(),
            conv3d(n_out, n_out),
        )
        self.skip = nn.Conv3d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


def FMRIEncoder(input_channels=16, latent_channels=8):
    """Encode [B, 16, 80, 96, 80] to [B, latent_channels, 10, 12, 10]."""
    return nn.Sequential(
        conv3d(input_channels, 32), Block3D(32, 32),
        conv3d(32, 64, stride=2, bias=False), Block3D(64, 64), Block3D(64, 64),
        conv3d(64, 64, stride=2, bias=False), Block3D(64, 64), Block3D(64, 64),
        conv3d(64, 64, stride=2, bias=False), Block3D(64, 64), Block3D(64, 64),
        conv3d(64, latent_channels),
    )


def FMRIDecoder(output_channels=16, latent_channels=8):
    """Decode [B, latent_channels, 10, 12, 10] back to [B, 16, 80, 96, 80]."""
    return nn.Sequential(
        Clamp(), conv3d(latent_channels, 64), nn.ReLU(),
        Block3D(64, 64), Block3D(64, 64),
        nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        conv3d(64, 64, bias=False),
        Block3D(64, 64), Block3D(64, 64),
        nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        conv3d(64, 64, bias=False),
        Block3D(64, 64), Block3D(64, 64),
        nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        conv3d(64, 32, bias=False),
        Block3D(32, 32),
        conv3d(32, output_channels),
    )


class FMRI3DAutoEncoder(nn.Module):
    """Trainable 3D autoencoder for fMRI latent diffusion.

    The original 2D TAESD checkpoints are not compatible with this module.
    Train this autoencoder on your fMRI volumes first, then train diffusion in
    the latent space produced by ``encode``.
    """

    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, input_channels=16, latent_channels=8):
        super().__init__()
        self.encoder = FMRIEncoder(input_channels=input_channels, latent_channels=latent_channels)
        self.decoder = FMRIDecoder(output_channels=input_channels, latent_channels=latent_channels)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1], useful only after latent statistics are calibrated."""
        return x.div(2 * FMRI3DAutoEncoder.latent_magnitude).add(FMRI3DAutoEncoder.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents."""
        return x.sub(FMRI3DAutoEncoder.latent_shift).mul(2 * FMRI3DAutoEncoder.latent_magnitude)


if __name__ == "__main__":
    model = FMRI3DAutoEncoder(input_channels=16, latent_channels=8)
    x = torch.randn(1, 16, 80, 96, 80)
    with torch.no_grad():
        z = model.encode(x)
        y = model.decode(z)
    print("input:", tuple(x.shape))
    print("latent:", tuple(z.shape))
    print("output:", tuple(y.shape))
