# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import torch
import math
from torch.nn import functional as F

def temporal_contrastive_loss(video, audio, temperature=0.1, exclusion_frames=4):
    if video.shape != audio.shape or video.ndim != 3 or min(video.shape) < 1:
        raise ValueError("Contrastive loss needs equal [batch, time, channels] tensors")
    if not video.is_floating_point() or not audio.is_floating_point() or video.device != audio.device or video.dtype != audio.dtype:
        raise ValueError("Contrastive features need matching floating-point types and devices")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Contrastive temperature must be finite and positive")
    if type(exclusion_frames) is not int or exclusion_frames < 0:
        raise ValueError("Excluded frame count must be a nonnegative integer")
    if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
        raise ValueError("Contrastive features must be finite")
    batch, time, channels = video.shape
    video_norm = F.normalize(video, dim=-1).reshape(-1, channels)
    audio_norm = F.normalize(audio, dim=-1).reshape(-1, channels)
    logits = video_norm @ audio_norm.T / temperature
    positions = torch.arange(batch * time, device=video.device)
    same_clip = positions[:, None] // time == positions[None, :] // time
    adjacent = (positions[:, None] % time - positions[None, :] % time).abs() <= exclusion_frames
    diagonal = torch.eye(batch * time, dtype=torch.bool, device=video.device)
    # Exclude nearby frames from the negative pairs.
    excluded = same_clip & adjacent & ~diagonal
    logits = logits.masked_fill(excluded, -10000)
    targets = torch.arange(batch * time, device=video.device)
    contrastive = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
    positive = 1 - (video_norm * audio_norm).sum(dim=-1).mean()
    video_std = torch.sqrt(video.var(dim=(0, 1), unbiased=False) + 1e-4)
    audio_std = torch.sqrt(audio.var(dim=(0, 1), unbiased=False) + 1e-4)
    variance = (F.relu(0.5 - video_std).mean() + F.relu(0.5 - audio_std).mean()) / 2
    total = contrastive + positive + variance
    return total, {"contrastive": float(contrastive.detach()), "positive": float(positive.detach()), "variance": float(variance.detach())}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
