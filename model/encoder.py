# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import torch
from torch import nn

class VideoEncoder(nn.Module):
    def __init__(self, dimension=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv3d(16, 32, 3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv3d(32, dimension, 3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(4, dimension), nn.SiLU(),
        )

    def forward(self, video):
        # Keep temporal resolution through spatial pooling.
        encoded = self.layers(video)
        return encoded.mean(dim=(-1, -2)).transpose(1, 2)

class AudioEncoder(nn.Module):
    def __init__(self, dimension=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(16, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, dimension, 3, padding=1), nn.GroupNorm(4, dimension), nn.SiLU(),
        )

    def forward(self, audio):
        encoded = self.layers(audio)
        return encoded.mean(dim=2).transpose(1, 2)

class EncoderPair(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.architecture == "compact":
            self.video = VideoEncoder(config.feature_dim)
            self.audio = AudioEncoder(config.feature_dim)
        elif config.architecture == "i3d_vggish":
            from model.vision import DenseI3D
            from model.sound import DenseVGGish
            self.video = DenseI3D(config.feature_dim)
            self.audio = DenseVGGish(config.feature_dim)
        else:
            raise ValueError("Unknown encoder architecture")

    def forward(self, video, audio):
        return self.video(video), self.audio(audio)

def device_from_name(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
