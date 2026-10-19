# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import numpy as np

class AlignmentError(ValueError):
    pass

def validate_times(values, name="timestamps"):
    times = np.asarray(values)
    if not np.issubdtype(times.dtype, np.number) or np.iscomplexobj(times):
        raise AlignmentError("Timestamps must be real numbers: " + name)
    times = times.astype(np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise AlignmentError("Invalid " + name)
    if not np.all(np.diff(times) > 0):
        raise AlignmentError("Timestamps must be strictly increasing: " + name)
    return times

def timestamp_dtw(video_times, audio_times, tolerance=0.05):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise AlignmentError("DTW tolerance must be positive and finite")
    video = validate_times(video_times, "video")
    audio = validate_times(audio_times, "audio")
    rows, columns = len(video), len(audio)
    if rows * columns > 4000000:
        raise AlignmentError("Apply DTW within short segments")
    cost = np.full((rows + 1, columns + 1), np.inf)
    parent = np.full((rows + 1, columns + 1), -1, dtype=np.int8)
    cost[0, 0] = 0.0
    for i in range(1, rows + 1):
        left = max(1, int(np.searchsorted(audio, video[i - 1] - tolerance)) + 1)
        right = min(columns, int(np.searchsorted(audio, video[i - 1] + tolerance, side="right")))
        for j in range(left, right + 1):
            candidates = (cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
            direction = int(np.argmin(candidates))
            cost[i, j] = candidates[direction] + abs(video[i - 1] - audio[j - 1])
            parent[i, j] = direction
    if not np.isfinite(cost[-1, -1]):
        raise AlignmentError("No timestamp DTW path inside the tolerance")
    pairs = []
    i, j = rows, columns
    while i > 0 and j > 0:
        pairs.append((i - 1, j - 1))
        direction = parent[i, j]
        if direction == 0:
            i, j = i - 1, j - 1
        elif direction == 1:
            i -= 1
        elif direction == 2:
            j -= 1
        else:
            raise AlignmentError("Broken DTW predecessor")
    if i or j:
        raise AlignmentError("Incomplete DTW path")
    pairs.reverse()
    # Warp timestamps only to preserve content mismatch.
    buckets = [[] for _ in range(rows)]
    for i, j in pairs:
        buckets[i].append(audio[j])
    corrected = np.array([(video[i] + np.mean(buckets[i])) / 2 for i in range(rows)])
    if not np.all(np.diff(corrected) > 0):
        raise AlignmentError("DTW produced nonmonotonic timestamps")
    return corrected, np.asarray(pairs, dtype=np.int64)

def interpolate_features(features, times, grid):
    values = np.asarray(features)
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise AlignmentError("Features must be real numbers")
    values = values.astype(np.float64)
    source = validate_times(times)
    target = validate_times(grid, "target")
    if values.ndim != 2 or len(values) != len(source) or not np.isfinite(values).all():
        raise AlignmentError("Features must have shape [time, channels]")
    if target[0] < source[0] - 1e-8 or target[-1] > source[-1] + 1e-8:
        raise AlignmentError("Interpolation cannot extrapolate missing observations")
    return np.column_stack([np.interp(target, source, column) for column in values.T])

def align_features(video, audio, video_times, audio_times, config):
    vt = validate_times(video_times, "video")
    at = validate_times(audio_times, "audio")
    video, audio = np.asarray(video), np.asarray(audio)
    if video.ndim != 2 or audio.ndim != 2 or video.shape[1] < 1 or audio.shape[1] < 1 or len(video) != len(vt) or len(audio) != len(at):
        raise AlignmentError("Feature matrices and timestamp counts disagree")
    for values in (video, audio):
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values) or not np.isfinite(values).all():
            raise AlignmentError("Feature values must be finite real numbers")
    if abs(vt[0] - at[0]) > config.alignment_tolerance:
        raise AlignmentError("Initial timestamps exceed the alignment tolerance")
    for times in (vt, at):
        if np.max(np.diff(times)) > config.maximum_gap:
            raise AlignmentError("Timestamp gap exceeds the evidence limit")
        native_rate = 1 / np.median(np.diff(times))
        if native_rate <= 2 * config.frequency_max:
            raise AlignmentError("Native feature cadence is below wavelet Nyquist rate")
    end = min(vt[-1], at[-1])
    video_mask = vt <= end + config.alignment_tolerance - 1e-9
    audio_mask = at <= end + config.alignment_tolerance - 1e-9
    video, vt = video[video_mask], vt[video_mask]
    audio, at = audio[audio_mask], at[audio_mask]
    corrected, pairs = timestamp_dtw(vt, at, config.alignment_tolerance)
    start = max(corrected[0], at[0])
    end = min(corrected[-1], at[-1])
    count = int(np.floor((end - start) * config.video_fps + 1e-9)) + 1
    if count < config.minimum_samples:
        raise AlignmentError("Insufficient overlapping features")
    grid = start + np.arange(count) / config.video_fps
    aligned_video = interpolate_features(video, corrected, grid)
    aligned_audio = interpolate_features(audio, at, grid)
    return aligned_video, aligned_audio, grid, {
        "dtw_pairs": len(pairs),
        "maximum_timestamp_adjustment": float(np.max(np.abs(corrected - vt))),
        "aligned_samples": count,
    }

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
