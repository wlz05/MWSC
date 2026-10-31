# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import numpy as np
from scipy.signal import fftconvolve

class InsufficientEvidence(ValueError):
    pass

def first_component(features, minimum_variance=1e-12):
    values = np.asarray(features)
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise InsufficientEvidence("Features must be real numbers")
    values = values.astype(np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or len(values) < 4 or not np.isfinite(values).all():
        raise InsufficientEvidence("Invalid feature matrix")
    with np.errstate(over="ignore", invalid="ignore"):
        centered = values - values.mean(axis=0, keepdims=True)
        variance = np.var(centered, axis=0)
    if not np.isfinite(centered).all() or not np.isfinite(variance).all():
        raise InsufficientEvidence("Feature magnitude exceeds the stable PCA range")
    if np.max(variance) <= minimum_variance:
        raise InsufficientEvidence("Feature sequence has no temporal variation")
    try:
        _, singular, right = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise InsufficientEvidence("PCA decomposition did not converge") from error
    direction = right[0]
    # Fix the PCA sign for a consistent temporal orientation.
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    series = centered @ direction
    with np.errstate(over="ignore", invalid="ignore"):
        deviation = float(np.std(series))
    if not np.isfinite(deviation) or not np.isfinite(singular).all():
        raise InsufficientEvidence("PCA output exceeds the stable numeric range")
    if deviation * deviation <= minimum_variance:
        raise InsufficientEvidence("First principal component has negligible variance")
    # Normalize singular values before squaring to avoid overflow.
    explained = float(1 / np.sum((singular / singular[0]) ** 2))
    return series / deviation, explained

def morlet_cwt(series, sample_rate, frequencies, omega=6.0):
    values = np.asarray(series, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
        raise InsufficientEvidence("Invalid wavelet input")
    if not np.isfinite(sample_rate) or sample_rate <= 0 or not np.isfinite(omega) or omega < 5:
        raise ValueError("Invalid wavelet sampling rate or central frequency")
    if frequencies.ndim != 1 or not np.all((frequencies > 0) & (frequencies < sample_rate / 2)):
        raise ValueError("Wavelet frequencies must lie below Nyquist")
    # Convert physical frequencies to sample-domain scales.
    scales = omega * sample_rate / (2 * np.pi * frequencies)
    coefficients = []
    boundary_fraction = []
    for scale in scales:
        radius = int(np.ceil(4 * scale))
        coordinate = np.arange(-radius, radius + 1) / scale
        kernel = np.pi ** (-0.25) * np.exp(-coordinate ** 2 / 2)
        kernel = kernel * (np.exp(1j * omega * coordinate) - np.exp(-omega ** 2 / 2)) / np.sqrt(scale)
        padded = np.pad(values, (radius, radius), mode="reflect")
        transformed = fftconvolve(padded, np.conj(kernel[::-1]), mode="same")
        coefficients.append(transformed[radius:radius + len(values)])
        cone_radius = np.sqrt(2) * scale
        boundary_fraction.append(min(1.0, 2 * cone_radius / len(values)))
    return np.asarray(coefficients), scales, np.asarray(boundary_fraction)

def scale_coherence(video_coefficients, audio_coefficients):
    video = np.asarray(video_coefficients, dtype=np.complex128)
    audio = np.asarray(audio_coefficients, dtype=np.complex128)
    if video.shape != audio.shape or video.ndim != 2 or min(video.shape) < 1 or not np.isfinite(video).all() or not np.isfinite(audio).all():
        raise ValueError("Wavelet coefficient shapes or values are invalid")
    video_scale = np.max(np.maximum(np.abs(video.real), np.abs(video.imag)), axis=1)[:, None]
    audio_scale = np.max(np.maximum(np.abs(audio.real), np.abs(audio.imag)), axis=1)[:, None]
    if np.any(video_scale == 0) or np.any(audio_scale == 0):
        raise InsufficientEvidence("A wavelet scale has zero energy")
    # Independent rescaling cancels from the coherence ratio.
    video = video.real / video_scale + 1j * (video.imag / video_scale)
    audio = audio.real / audio_scale + 1j * (audio.imag / audio_scale)
    # Square the magnitude after the temporal sum.
    numerator = np.abs(np.sum(video * np.conj(audio), axis=1)) ** 2
    denominator = np.sum(np.abs(video) ** 2, axis=1) * np.sum(np.abs(audio) ** 2, axis=1)
    if np.any(denominator <= np.finfo(np.float64).tiny):
        raise InsufficientEvidence("A wavelet scale has zero energy")
    return np.clip(numerator / denominator, 0, 1)

def gaussian_weights(frequencies, mean=4.0, std=1.5, mode="patent_raw"):
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if frequencies.ndim != 1 or not len(frequencies) or std <= 0 or not np.isfinite(std) or not np.isfinite(mean) or not np.isfinite(frequencies).all():
        raise ValueError("Invalid Gaussian weights")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        logweights = -0.5 * ((frequencies - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)
    if mode == "normalized":
        if not np.isfinite(logweights).any():
            raise ValueError("Gaussian grid has no representable weights")
        # Shift log weights before exponentiation to avoid underflow.
        weights = np.exp(logweights - np.max(logweights))
        weights = weights / weights.sum()
    elif mode == "patent_raw":
        weights = np.exp(logweights)
    else:
        raise ValueError("Unknown Gaussian weight mode")
    if not np.isfinite(weights).all() or weights.sum() <= np.finfo(np.float64).tiny:
        raise ValueError("Gaussian grid has no usable weight mass")
    return weights

def synchronization_score(video_features, audio_features, config):
    if len(video_features) != len(audio_features) or len(video_features) < config.minimum_samples:
        raise InsufficientEvidence("Aligned sequences are too short or unequal")
    video, video_explained = first_component(video_features, config.minimum_feature_variance)
    audio, audio_explained = first_component(audio_features, config.minimum_feature_variance)
    frequencies = np.linspace(config.frequency_min, config.frequency_max, config.frequency_count)
    video_wave, scales, boundary = morlet_cwt(video, config.video_fps, frequencies, config.morlet_omega)
    audio_wave, _, _ = morlet_cwt(audio, config.video_fps, frequencies, config.morlet_omega)
    coherence = scale_coherence(video_wave, audio_wave)
    weights = gaussian_weights(frequencies, config.gaussian_mean, config.gaussian_std, config.weight_mode)
    score = float(np.dot(weights, coherence))
    return {
        "synchronization_score": score,
        "score_upper_bound": float(weights.sum()),
        "frequencies_hz": frequencies.tolist(),
        "wavelet_scales_samples": scales.tolist(),
        "scale_coherence": coherence.tolist(),
        "gaussian_weights": weights.tolist(),
        "weight_mode": config.weight_mode,
        "video_pca_explained_variance": video_explained,
        "audio_pca_explained_variance": audio_explained,
        "boundary_affected_fraction": boundary.tolist(),
        "warnings": ["Short windows provide limited evidence at low frequencies", "Synchronization score is not a fake probability"],
    }

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
