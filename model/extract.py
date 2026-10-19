# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import numpy as np
import torch
from core.audio import stft_features, logmel_from_stft
from core.wavelet import InsufficientEvidence

def prepare_inputs(clip, config):
    try:
        clip.validate()
    except ValueError as error:
        raise InsufficientEvidence(str(error)) from error
    if clip.sample_rate != config.sample_rate:
        raise ValueError("Audio must be resampled before feature extraction")
    if len(clip.video_times) < 2 or len(clip.frames) != len(clip.video_times) or not len(clip.waveform):
        raise InsufficientEvidence("Media contains too few matching observations")
    if not np.isfinite(clip.video_times).all() or not np.isfinite(clip.waveform).all() or not np.all(np.diff(clip.video_times) > 0):
        raise InsufficientEvidence("Media observations or timestamps are invalid")
    if 1 / np.median(np.diff(clip.video_times)) <= 2 * config.frequency_max:
        raise InsufficientEvidence("Native video cadence cannot resolve the wavelet frequency range")
    if float(np.sqrt(np.mean(clip.waveform.astype(np.float64) ** 2))) < config.minimum_rms:
        raise InsufficientEvidence("Audio has insufficient signal energy")
    if np.max(np.diff(clip.video_times)) > config.maximum_gap:
        raise InsufficientEvidence("Video contains an excessive frame gap")
    step = 1 / config.video_fps
    end = clip.video_times[-1]
    count = int(np.floor((end - clip.video_times[0]) / step + 1e-6)) + 1
    grid = clip.video_times[0] + np.arange(count) * step
    # Select observed frames without synthesizing missing video content.
    right = np.searchsorted(clip.video_times, grid).clip(0, len(clip.frames) - 1)
    left = np.maximum(right - 1, 0)
    closest = np.where(np.abs(clip.video_times[left] - grid) <= np.abs(clip.video_times[right] - grid), left, right)
    if np.any(np.abs(clip.video_times[closest] - grid) > config.alignment_tolerance):
        raise InsufficientEvidence("Frame sampling exceeds timestamp tolerance")
    frames = clip.frames[closest].astype(np.float32) / 255
    if len(frames) < config.minimum_samples:
        raise InsufficientEvidence("Too few video frames")
    spectrum, audio_times = stft_features(clip.waveform, config)
    if config.architecture == "i3d_vggish":
        spectrum = logmel_from_stft(spectrum, config)
    video = torch.from_numpy(np.ascontiguousarray(frames))[None, None]
    audio = torch.from_numpy(np.ascontiguousarray(spectrum))[None, None]
    return video, audio, grid, clip.audio_start + audio_times

class NeuralExtractor:
    def __init__(self, model, config, device=None):
        self.model = model.eval()
        self.config = config.validate()
        self.device = device if device is not None else next(model.parameters()).device

    def extract(self, clip):
        video, audio, video_times, audio_times = prepare_inputs(clip, self.config)
        with torch.inference_mode():
            video_encoded, audio_encoded = self.model(video.to(self.device), audio.to(self.device))
        video_features = video_encoded[0].detach().cpu().float().numpy()
        audio_features = audio_encoded[0].detach().cpu().float().numpy()
        if video_features.shape[0] != len(video_times) or audio_features.shape[0] != len(audio_times):
            raise ValueError("Encoders must preserve dense temporal resolution")
        if video_features.ndim != 2 or audio_features.ndim != 2:
            raise ValueError("Encoder output requires shape [batch, time, channels]")
        return video_features, audio_features, video_times, audio_times

class ScriptedPair(torch.nn.Module):
    def __init__(self, video_path, audio_path, device="cpu"):
        super().__init__()
        from support.storage import relative_path
        self.video = torch.jit.load(str(relative_path(video_path, True)), map_location=device)
        self.audio = torch.jit.load(str(relative_path(audio_path, True)), map_location=device)

    def forward(self, video, audio):
        return self.video(video), self.audio(audio)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
