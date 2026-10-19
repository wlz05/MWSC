# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import math
from numbers import Integral
import numpy as np
from scipy.signal import resample_poly

def mono_resample(waveform, source_rate, target_rate):
    if any(isinstance(rate, bool) or not isinstance(rate, Integral) or rate <= 0 for rate in (source_rate, target_rate)):
        raise ValueError("Audio sampling rates require positive integers")
    values = np.asarray(waveform)
    if values.ndim not in (1, 2) or not values.size:
        raise ValueError("Audio must have shape [samples] or [samples, channels]")
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        if info.min == 0:
            # Center unsigned PCM before scaling its amplitude.
            values = (values.astype(np.float64) - (info.max + 1) / 2) / ((info.max + 1) / 2)
        else:
            values = values.astype(np.float64) / max(abs(info.min), abs(info.max))
    else:
        values = values.astype(np.float64)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if source_rate <= 0 or target_rate <= 0 or not np.isfinite(values).all():
        raise ValueError("Invalid audio sampling rate or values")
    if source_rate != target_rate:
        divisor = math.gcd(int(source_rate), int(target_rate))
        values = resample_poly(values, target_rate // divisor, source_rate // divisor)
    return values.astype(np.float32)

def preemphasize(waveform, coefficient=0.97):
    if not np.isfinite(coefficient) or not 0 <= coefficient < 1:
        raise ValueError("Invalid pre-emphasis coefficient")
    values = np.asarray(waveform, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid mono waveform")
    result = values.copy()
    result[1:] -= coefficient * values[:-1]
    return result

def stft_features(waveform, config):
    values = preemphasize(waveform, config.preemphasis)
    length = round(config.sample_rate * config.frame_ms / 1000)
    hop = round(config.sample_rate * config.hop_ms / 1000)
    if len(values) < length:
        raise ValueError("Audio is shorter than one analysis frame")
    frames = np.lib.stride_tricks.sliding_window_view(values, length)[::hop]
    # Use a periodic Hann window for the short-time spectrum.
    window = np.hanning(length + 1)[:-1]
    spectrum = np.fft.rfft(frames * window, n=config.n_fft, axis=1)
    power = (np.abs(spectrum) ** 2) / max(float(np.sum(window ** 2)), 1e-12)
    times = (np.arange(len(frames)) * hop + (length - 1) / 2) / config.sample_rate
    return np.log1p(power).T.astype(np.float32), times

def mel_filterbank(sample_rate, n_fft, bins=64, lower=125.0, upper=7500.0):
    upper = min(upper, sample_rate / 2)
    if not 0 <= lower < upper:
        raise ValueError("Invalid mel filter limits")
    to_mel = lambda hz: 2595 * np.log10(1 + np.asarray(hz) / 700)
    to_hz = lambda mel: 700 * (10 ** (np.asarray(mel) / 2595) - 1)
    points = to_hz(np.linspace(to_mel(lower), to_mel(upper), bins + 2))
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    rising = (frequencies[:, None] - points[:-2]) / (points[1:-1] - points[:-2])
    falling = (points[2:] - frequencies[:, None]) / (points[2:] - points[1:-1])
    return np.maximum(0, np.minimum(rising, falling)).astype(np.float32)

def logmel_from_stft(log_power, config):
    filters = mel_filterbank(config.sample_rate, config.n_fft)
    power = np.expm1(np.asarray(log_power, dtype=np.float64))
    return np.log(filters.T @ power + 0.01).astype(np.float32)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
