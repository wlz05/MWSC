# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import math
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from core.media import read_media, probe_duration
from dataset.manifest import select_rows
from model.extract import prepare_inputs

class RealSegmentDataset(Dataset):
    def __init__(self, rows, config, split="train", seed=42, random_crop=True):
        self.rows = select_rows(rows, split, real_only=True)
        self.config = config.validate()
        self.seed = int(seed)
        self.epoch = 0
        self.random_crop = random_crop
        self.durations = {}

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        row = self.rows[index]
        rng = np.random.default_rng(self.seed + index + self.epoch * len(self.rows))
        if index not in self.durations:
            available = probe_duration(row["video"])
            if row["duration"] != "" and abs(float(row["duration"]) - available) > 0.1:
                raise ValueError("Manifest duration differs from decoded video duration")
            self.durations[index] = available
        available = self.durations[index]
        if available + 0.001 < self.config.segment_seconds:
            raise ValueError("Training sample is shorter than one segment: " + row["video"])
        maximum = max(available - self.config.segment_seconds, 0)
        start = float(rng.uniform(0, maximum)) if self.random_crop and maximum > 0 else maximum / 2
        clip = read_media(row["video"], self.config, row["audio"] or None, start, self.config.segment_seconds, row["audio_offset"])
        video, audio, vt, at = prepare_inputs(clip, self.config)
        start_time, end_time = max(vt[0], at[0]), min(vt[-1], at[-1])
        grid = np.linspace(start_time, end_time, round(self.config.segment_seconds * self.config.video_fps) - 1)
        if end_time - start_time < self.config.segment_seconds * 0.8:
            raise ValueError("Training interval has insufficient overlap")
        # Relative times preserve float32 precision at large PTS origins.
        return {"video": video[0], "audio": audio[0], "video_times": torch.tensor(vt - start_time, dtype=torch.float32), "audio_times": torch.tensor(at - start_time, dtype=torch.float32), "grid": torch.tensor(grid - start_time, dtype=torch.float32), "source": row["source"]}

def collate_segments(samples):
    # Keep original timestamps to trim padded features after encoding.
    video_length = max(sample["video"].shape[1] for sample in samples)
    audio_length = max(sample["audio"].shape[-1] for sample in samples)
    video = torch.stack([F.pad(sample["video"], (0, 0, 0, 0, 0, video_length - sample["video"].shape[1]), mode="replicate") for sample in samples])
    audio = torch.stack([F.pad(sample["audio"], (0, audio_length - sample["audio"].shape[-1]), mode="replicate") for sample in samples])
    return {"video": video, "audio": audio, "samples": samples}

def temporal_interpolate(values, source_times, target_times):
    source_times = source_times.to(values.device)
    target_times = target_times.to(values.device)
    if target_times[0] < source_times[0] - 1e-5 or target_times[-1] > source_times[-1] + 1e-5:
        raise ValueError("Training interpolation cannot extrapolate")
    right = torch.searchsorted(source_times.contiguous(), target_times.contiguous()).clamp(1, len(source_times) - 1)
    left = right - 1
    weight = (target_times - source_times[left]) / (source_times[right] - source_times[left])
    return values[left] * (1 - weight[:, None]) + values[right] * weight[:, None]

def aligned_batch(video, audio, samples):
    aligned_video, aligned_audio = [], []
    for index, sample in enumerate(samples):
        vt, at, grid = sample["video_times"], sample["audio_times"], sample["grid"]
        aligned_video.append(temporal_interpolate(video[index, :len(vt)], vt, grid))
        aligned_audio.append(temporal_interpolate(audio[index, :len(at)], at, grid))
    return torch.stack(aligned_video), torch.stack(aligned_audio)

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
