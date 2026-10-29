# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math

@dataclass(frozen=True)
class Config:
    sample_rate: int = 16000
    video_fps: float = 30.0
    image_size: int = 224
    preemphasis: float = 0.97
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    segment_seconds: float = 2.0
    overlap: float = 0.5
    alignment_tolerance: float = 0.05
    frequency_min: float = 0.5
    frequency_max: float = 4.0
    frequency_count: int = 8
    morlet_omega: float = 6.0
    gaussian_mean: float = 4.0
    gaussian_std: float = 1.5
    weight_mode: str = "patent_raw"
    threshold: float = 0.75
    history_size: int = 100
    ema_alpha: float = 0.9
    tolerance_k: float = 2.0
    minimum_samples: int = 16
    minimum_rms: float = 0.00001
    minimum_feature_variance: float = 1e-12
    maximum_gap: float = 0.15
    maximum_clip_seconds: float = 60.0
    feature_dim: int = 64
    architecture: str = "compact"

    def validate(self):
        integer_keys = ("sample_rate", "image_size", "n_fft", "frequency_count", "history_size", "minimum_samples", "feature_dim")
        for key in integer_keys:
            if isinstance(getattr(self, key), bool) or not isinstance(getattr(self, key), int):
                raise ValueError("Configuration requires an integer: " + key)
        for key, value in asdict(self).items():
            if key not in ("weight_mode", "architecture") and (isinstance(value, bool) or not isinstance(value, (float, int))):
                raise ValueError("Configuration requires a number: " + key)
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError("Nonfinite configuration: " + key)
        if self.sample_rate < 8000 or self.video_fps <= 2 * self.frequency_max:
            raise ValueError("Sampling rates cannot resolve the frequency range")
        if self.image_size < 32 or self.feature_dim < 8:
            raise ValueError("Image size or feature dimension is too small")
        if self.architecture == "compact" and self.feature_dim % 4:
            raise ValueError("Compact feature dimension must be divisible by four")
        if self.segment_seconds <= 0 or not 0 <= self.overlap < 1:
            raise ValueError("Invalid segmentation parameters")
        if not 0 <= self.preemphasis < 1 or self.frame_ms <= self.hop_ms or self.hop_ms <= 0:
            raise ValueError("Invalid audio parameters")
        if round(self.sample_rate * self.hop_ms / 1000) < 1 or round(self.sample_rate * self.frame_ms / 1000) < 2:
            raise ValueError("Audio windows and hops must contain actual samples")
        if round(self.segment_seconds * self.video_fps) < self.minimum_samples:
            raise ValueError("Segment cannot provide the required video samples")
        if self.segment_seconds * (1 - self.overlap) * self.video_fps < 1 - 1e-9:
            raise ValueError("Window step must cover at least one video frame")
        if self.n_fft < round(self.sample_rate * self.frame_ms / 1000):
            raise ValueError("FFT size is shorter than the analysis frame")
        if not 0 < self.frequency_min <= self.frequency_max or self.frequency_count < 2:
            raise ValueError("Invalid wavelet frequencies")
        if self.gaussian_std <= 0 or self.morlet_omega < 5:
            raise ValueError("Invalid wavelet or Gaussian parameter")
        if self.weight_mode not in ("patent_raw", "normalized"):
            raise ValueError("Unknown Gaussian weight mode")
        if self.history_size < 1 or not 0 <= self.ema_alpha < 1 or self.tolerance_k < 0:
            raise ValueError("Invalid adaptive threshold parameters")
        if self.alignment_tolerance <= 0 or self.maximum_gap <= 0:
            raise ValueError("Invalid timestamp tolerance")
        if self.minimum_samples < 4 or self.minimum_rms < 0 or self.minimum_feature_variance <= 0:
            raise ValueError("Invalid evidence quality parameters")
        if self.maximum_clip_seconds < self.segment_seconds:
            raise ValueError("Clip limit is shorter than one segment")
        if self.architecture not in ("compact", "i3d_vggish"):
            raise ValueError("Unknown encoder architecture")
        return self

    def to_dict(self):
        return asdict(self)

    def signature(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("Configuration requires a dictionary")
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError("Unknown configuration keys: " + ", ".join(sorted(unknown)))
        return cls(**values).validate()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
