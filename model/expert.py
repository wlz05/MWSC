# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import pickle
# Copyright (c) 2016-present Joon Son Chung.
# See README.md for the SyncNet MIT license and attribution.
import torch
from torch import nn
from model.encoder import device_from_name
from support.storage import relative_path

class SyncNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.netcnnaud = nn.Sequential(*self.audio_layers())
        self.netcnnlip = nn.Sequential(*self.video_layers())
        self.netfcaud = self.projection()
        self.netfclip = self.projection()

    @staticmethod
    def projection():
        return nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 1024))

    @staticmethod
    def audio_layers():
        layers = []
        for source, target, kernel, padding, pool in ((1, 64, (3, 3), (1, 1), ((1, 1), (1, 1))), (64, 192, (3, 3), (1, 1), ((3, 3), (1, 2))), (192, 384, (3, 3), (1, 1), None), (384, 256, (3, 3), (1, 1), None), (256, 256, (3, 3), (1, 1), ((3, 3), (2, 2))), (256, 512, (5, 4), (0, 0), None)):
            layers.extend((nn.Conv2d(source, target, kernel, padding=padding), nn.BatchNorm2d(target), nn.ReLU()))
            if pool is not None:
                layers.append(nn.MaxPool2d(*pool))
        return layers

    @staticmethod
    def video_layers():
        layers = []
        settings = ((3, 96, (5, 7, 7), (1, 2, 2), 0, ((1, 3, 3), (1, 2, 2), 0)), (96, 256, (1, 5, 5), (1, 2, 2), (0, 1, 1), ((1, 3, 3), (1, 2, 2), (0, 1, 1))), (256, 256, (1, 3, 3), 1, (0, 1, 1), None), (256, 256, (1, 3, 3), 1, (0, 1, 1), None), (256, 256, (1, 3, 3), 1, (0, 1, 1), ((1, 3, 3), (1, 2, 2), 0)), (256, 512, (1, 6, 6), 1, 0, None))
        for source, target, kernel, stride, padding, pool in settings:
            layers.extend((nn.Conv3d(source, target, kernel, stride=stride, padding=padding), nn.BatchNorm3d(target), nn.ReLU()))
            if pool is not None:
                layers.append(nn.MaxPool3d(*pool))
        return layers

    def forward_lip(self, values):
        return self.netfclip(self.netcnnlip(values).flatten(1))

    def forward_aud(self, values):
        return self.netfcaud(self.netcnnaud(values).flatten(1))

def load_expert(path, device="auto"):
    try:
        state = torch.load(relative_path(path, True), map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as error:
        raise ValueError("SyncNet requires a safe pretrained tensor state dictionary") from error
    if not isinstance(state, dict) or not state or not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("SyncNet checkpoint must contain named tensors")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("SyncNet checkpoint contains nonfinite tensors")
    model = SyncNet()
    expected = model.state_dict()
    missing = set(expected) - set(state)
    if set(state) - set(expected) or any(not name.endswith("num_batches_tracked") for name in missing):
        raise ValueError("SyncNet checkpoint keys do not match the v2 expert")
    if any(expected[name].shape != value.shape for name, value in state.items()):
        raise ValueError("SyncNet checkpoint tensor shapes do not match the v2 expert")
    model.load_state_dict({**expected, **state}, strict=True)
    return model.to(device_from_name(device)).eval()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
