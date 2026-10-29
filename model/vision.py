# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import torch
from torch import nn
from torch.nn import functional as F

class SamePad3D(nn.Module):
    def __init__(self, kernel, stride):
        super().__init__()
        self.kernel = tuple(kernel)
        self.stride = tuple(stride)

    def forward(self, tensor):
        padding = []
        for size, kernel, stride in zip(reversed(tensor.shape[-3:]), reversed(self.kernel), reversed(self.stride)):
            total = max((size + stride - 1) // stride * stride + kernel - stride - size, 0)
            padding.extend((total // 2, total - total // 2))
        return F.pad(tensor, padding)

class Unit3D(nn.Module):
    def __init__(self, incoming, outgoing, kernel=(1, 1, 1), stride=(1, 1, 1)):
        super().__init__()
        self.pad = SamePad3D(kernel, stride)
        self.conv3d = nn.Conv3d(incoming, outgoing, kernel, stride=stride, bias=False)
        self.bn = nn.BatchNorm3d(outgoing, eps=0.001, momentum=0.01)

    def forward(self, tensor):
        return F.relu(self.bn(self.conv3d(self.pad(tensor))), inplace=False)

class SamePool3D(nn.Module):
    def __init__(self, kernel=(3, 3, 3), stride=(1, 2, 2)):
        super().__init__()
        self.pad = SamePad3D(kernel, stride)
        self.pool = nn.MaxPool3d(kernel, stride=stride)

    def forward(self, tensor):
        return self.pool(self.pad(tensor))

class Inception3D(nn.Module):
    def __init__(self, incoming, channels):
        super().__init__()
        self.b0 = Unit3D(incoming, channels[0])
        self.b1a = Unit3D(incoming, channels[1])
        self.b1b = Unit3D(channels[1], channels[2], (3, 3, 3))
        self.b2a = Unit3D(incoming, channels[3])
        self.b2b = Unit3D(channels[3], channels[4], (3, 3, 3))
        self.b3a = SamePool3D((3, 3, 3), (1, 1, 1))
        self.b3b = Unit3D(incoming, channels[5])

    def forward(self, tensor):
        return torch.cat((self.b0(tensor), self.b1b(self.b1a(tensor)), self.b2b(self.b2a(tensor)), self.b3b(self.b3a(tensor))), dim=1)

class DenseI3D(nn.Module):
    def __init__(self, dimension=64):
        super().__init__()
        # Dense temporal strides retain speech rhythm bandwidth.
        self.end_points = nn.ModuleDict({
            "Conv3d_1a_7x7": Unit3D(3, 64, (7, 7, 7), (1, 2, 2)),
            "MaxPool3d_2a_3x3": SamePool3D((1, 3, 3)),
            "Conv3d_2b_1x1": Unit3D(64, 64),
            "Conv3d_2c_3x3": Unit3D(64, 192, (3, 3, 3)),
            "MaxPool3d_3a_3x3": SamePool3D((1, 3, 3)),
            "Mixed_3b": Inception3D(192, (64, 96, 128, 16, 32, 32)),
            "Mixed_3c": Inception3D(256, (128, 128, 192, 32, 96, 64)),
            "MaxPool3d_4a_3x3": SamePool3D(),
            "Mixed_4b": Inception3D(480, (192, 96, 208, 16, 48, 64)),
            "Mixed_4c": Inception3D(512, (160, 112, 224, 24, 64, 64)),
            "Mixed_4d": Inception3D(512, (128, 128, 256, 24, 64, 64)),
            "Mixed_4e": Inception3D(512, (112, 144, 288, 32, 64, 64)),
            "Mixed_4f": Inception3D(528, (256, 160, 320, 32, 128, 128)),
            "MaxPool3d_5a_2x2": SamePool3D((2, 2, 2), (1, 2, 2)),
            "Mixed_5b": Inception3D(832, (256, 160, 320, 32, 128, 128)),
            "Mixed_5c": Inception3D(832, (384, 192, 384, 48, 128, 128)),
        })
        self.projection = nn.Linear(1024, dimension)

    def forward(self, video):
        values = video.repeat(1, 3, 1, 1, 1) * 2 - 1
        for layer in self.end_points.values():
            values = layer(values)
        temporal = values.mean(dim=(-1, -2)).transpose(1, 2)
        return self.projection(temporal)

    def import_backbone(self, state):
        target = self.end_points.state_dict()
        converted = {}
        for key, value in state.items():
            name = key.removeprefix("module.").removeprefix("end_points.")
            if name in target:
                converted[name] = value
        missing = set(target) - set(converted)
        required = {key for key in missing if not key.endswith("num_batches_tracked")}
        if required:
            raise ValueError("Incomplete I3D backbone weights: " + ", ".join(sorted(required)[:8]))
        for key in missing:
            converted[key] = target[key]
        self.end_points.load_state_dict(converted, strict=True)
        return {"loaded": len(converted), "ignored": len(state) - len(converted)}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
