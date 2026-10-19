# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import torch
from torch.nn import functional as F

def temporal_contrastive_loss(video, audio, temperature=0.1, exclusion_frames=4):
    if video.shape != audio.shape or video.ndim != 3:
        raise ValueError("Contrastive loss needs equal [batch, time, channels] tensors")
    batch, time, channels = video.shape
    video_norm = F.normalize(video, dim=-1).reshape(-1, channels)
    audio_norm = F.normalize(audio, dim=-1).reshape(-1, channels)
    logits = video_norm @ audio_norm.T / temperature
    positions = torch.arange(batch * time, device=video.device)
    same_clip = positions[:, None] // time == positions[None, :] // time
    adjacent = (positions[:, None] % time - positions[None, :] % time).abs() <= exclusion_frames
    diagonal = torch.eye(batch * time, dtype=torch.bool, device=video.device)
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

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
