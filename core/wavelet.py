# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import numpy as np
from scipy.signal import fftconvolve

class InsufficientEvidence(ValueError):
    pass

def first_component(features, minimum_variance=1e-12):
    # Xc = X - mean_t(X); z = Xc @ r1 / std(Xc @ r1).
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
    if frequencies.ndim != 1 or not len(frequencies) or not np.all((frequencies > 0) & (frequencies < sample_rate / 2)):
        raise ValueError("Wavelet frequencies must lie below Nyquist")
    # Convert physical frequencies to sample-domain scales.
    # s_j = omega0 * Fs / (2*pi*f_j), in samples.
    scales = omega * sample_rate / (2 * np.pi * frequencies)
    coefficients = []
    boundary_fraction = []
    for scale in scales:
        radius = int(np.ceil(4 * scale))
        coordinate = np.arange(-radius, radius + 1) / scale
        # psi(u) = pi^(-1/4) exp(-u^2/2) [exp(i*omega0*u)-exp(-omega0^2/2)].
        kernel = np.pi ** (-0.25) * np.exp(-coordinate ** 2 / 2)
        kernel = kernel * (np.exp(1j * omega * coordinate) - np.exp(-omega ** 2 / 2)) / np.sqrt(scale)
        padded = np.pad(values, (radius, radius), mode="reflect")
        transformed = fftconvolve(padded, np.conj(kernel[::-1]), mode="same")
        coefficients.append(transformed[radius:radius + len(values)])
        cone_radius = np.sqrt(2) * scale
        boundary_fraction.append(min(1.0, 2 * cone_radius / len(values)))
    return np.asarray(coefficients), scales, np.asarray(boundary_fraction)

def coherence_terms(video_coefficients, audio_coefficients):
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
    cross = video * np.conj(audio)
    denominator = np.sum(np.abs(video) ** 2, axis=1) * np.sum(np.abs(audio) ** 2, axis=1)
    if np.any(denominator <= np.finfo(np.float64).tiny):
        raise InsufficientEvidence("A wavelet scale has zero energy")
    return cross, denominator

def scale_coherence(video_coefficients, audio_coefficients):
    cross, denominator = coherence_terms(video_coefficients, audio_coefficients)
    # C_j = |sum_t Wv_j(t)*conj(Wa_j(t))|^2 / (Ev_j*Ea_j).
    numerator = np.abs(np.sum(cross, axis=1)) ** 2
    return np.clip(numerator / denominator, 0, 1)

def coherence_evidence(video_coefficients, audio_coefficients, weights):
    cross, denominator = coherence_terms(video_coefficients, audio_coefficients)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(cross),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Evidence weights must be finite, nonnegative, and match the scales")
    mass = float(weights.sum())
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Evidence weights require positive finite mass")
    resultant = cross.sum(axis=1)
    raw = np.abs(resultant) ** 2 / denominator
    coherence = np.clip(raw, 0, 1)
    # e_j(t) = w_j * Re[c_j(t)*conj(sum_u c_j(u))] / (Ev_j*Ea_j).
    terms = np.real(cross * resultant.conj()[:, None]) / denominator[:, None]
    terms *= weights[:, None]
    # Preserve the scored clipping convention at floating-point boundaries.
    correction = np.divide(coherence, raw, out=np.ones_like(raw), where=raw > 0)
    terms *= correction[:, None]
    contributions = weights * coherence
    score = float(contributions.sum())
    temporal = terms.sum(axis=0)
    rho = resultant / np.sqrt(denominator)
    phase = [float(np.angle(value)) if abs(value) > np.finfo(np.float64).eps else None for value in rho]
    return {"scale_contributions": contributions.tolist(), "cross_phase_rad": phase,
        "time_frequency_contributions": terms.tolist(), "time_contributions": temporal.tolist(),
        "reconstructed_score": float(temporal.sum()), "conservation_error": float(temporal.sum()-score),
        "definition": "Signed score contributions; sum over time and scale equals MSSC"}

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

def synchronization_score(video_features, audio_features, config, explain=False):
    if type(explain) is not bool:
        raise ValueError("Explain must be Boolean")
    config.validate()
    if len(video_features) != len(audio_features) or len(video_features) < config.minimum_samples:
        raise InsufficientEvidence("Aligned sequences are too short or unequal")
    video, video_explained = first_component(video_features, config.minimum_feature_variance)
    audio, audio_explained = first_component(audio_features, config.minimum_feature_variance)
    frequencies = np.linspace(config.frequency_min, config.frequency_max, config.frequency_count)
    video_wave, scales, boundary = morlet_cwt(video, config.video_fps, frequencies, config.morlet_omega)
    audio_wave, _, _ = morlet_cwt(audio, config.video_fps, frequencies, config.morlet_omega)
    coherence = scale_coherence(video_wave, audio_wave)
    weights = gaussian_weights(frequencies, config.gaussian_mean, config.gaussian_std, config.weight_mode)
    # MSSC = sum_j w_j*C_j; its normalized form divides by sum_j w_j.
    contributions = weights * coherence
    score = float(np.dot(weights, coherence))
    result = {
        "synchronization_score": score,
        "normalized_synchronization_score": score / float(weights.sum()),
        "scale_contributions": contributions.tolist(),
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
    if explain:
        result["explanation"] = coherence_evidence(video_wave, audio_wave, weights)
        distance = np.minimum(np.arange(len(video)), len(video)-1-np.arange(len(video)))
        result["explanation"]["interior_support_mask"] = (distance[None, :] >= np.sqrt(2)*scales[:, None]).tolist()
    return result

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
