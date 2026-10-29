# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from torch import nn

class DenseVGGish(nn.Module):
    def __init__(self, dimension=64):
        super().__init__()
        # Pool frequency only to preserve dense timing.
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
        )
        self.projection = nn.Linear(512, dimension)

    def forward(self, spectrogram):
        values = self.features(spectrogram).mean(dim=2).transpose(1, 2)
        return self.projection(values)

    def import_backbone(self, state):
        target = self.features.state_dict()
        converted = {}
        for key, value in state.items():
            name = key.removeprefix("module.").removeprefix("features.")
            if name in target:
                converted[name] = value
        if set(converted) != set(target):
            missing = set(target) - set(converted)
            raise ValueError("Incomplete VGGish convolution weights: " + ", ".join(sorted(missing)[:8]))
        self.features.load_state_dict(converted, strict=True)
        return {"loaded": len(converted), "ignored": len(state) - len(converted)}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
