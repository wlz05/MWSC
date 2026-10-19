# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from dataclasses import replace
import importlib.util
import math
import numpy as np
import torch
from scipy.fft import dct
from config import Config
from core.audio import preemphasize
from core.face import FaceReader
from core.media import read_media, probe_duration, probe_stream_origins
from engine.detect import checkpoint_identity
from model.expert import load_expert
from support.storage import atomic_json, relative_path, paths_match

DEFAULT_WEIGHTS = "../runtime/mwsc/syncnet.model"

def mfcc_features(waveform):
    raw = np.asarray(waveform)
    if raw.ndim != 1 or not len(raw) or not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw) or not np.isfinite(raw).all():
        raise ValueError("MFCC input requires finite real mono audio")
    pcm = np.rint(np.clip(raw, -1, 32767 / 32768) * 32768)
    signal = preemphasize(pcm)
    count = max(1, 1 + int(np.ceil((len(signal) - 400) / 160)))
    signal = np.pad(signal, (0, max(0, (count - 1) * 160 + 400 - len(signal))))
    frames = np.lib.stride_tricks.sliding_window_view(signal, 400)[::160]
    power = np.abs(np.fft.rfft(frames, n=512, axis=1)) ** 2 / 512
    energy = np.maximum(power.sum(axis=1), np.finfo(float).eps)
    mel = np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 28)
    points = np.floor(513 * 700 * (10 ** (mel / 2595) - 1) / 16000).astype(int)
    filters = np.zeros((26, 257))
    for index, (left, center, right) in enumerate(zip(points, points[1:], points[2:])):
        filters[index, left:center] = np.arange(center - left) / (center - left)
        filters[index, center:right] = np.arange(right - center, 0, -1) / (right - center)
    cepstra = dct(np.log(np.maximum(power @ filters.T, np.finfo(float).eps)), type=2, axis=1, norm="ortho")[:, :13]
    cepstra *= 1 + 11 * np.sin(np.pi * np.arange(13) / 22)
    cepstra[:, 0] = np.log(energy)
    return cepstra.T.astype(np.float32)

def profile_summary(curve, vshift, minimum_confidence=3.0, tolerance=0.08):
    if type(vshift) is not int or not 1 <= vshift <= 50 or not np.isfinite([minimum_confidence, tolerance]).all() or minimum_confidence < 0 or tolerance < 0:
        raise ValueError("Invalid SyncNet profile options")
    curve = np.asarray(curve, dtype=np.float64)
    if curve.shape != (2 * vshift + 1,) or not np.isfinite(curve).all():
        raise ValueError("SyncNet distance curve is invalid")
    index = int(np.argmin(curve))
    distance = float(curve[index])
    confidence = float(np.median(curve) - distance)
    offset = vshift - index
    audio_lag = -offset / 25
    return {"lse_c": confidence, "lse_d": distance, "zero_offset_distance": float(curve[vshift]), "syncnet_offset_frames": offset, "audio_lag_seconds": audio_lag, "audio_lag_ms": audio_lag * 1000, "sync_decision": "uncertain" if confidence < minimum_confidence else "synchronized" if abs(audio_lag) <= tolerance + 1e-9 else "out_of_sync", "search_boundary_hit": index in (0, 2 * vshift), "distance_curve": curve.tolist()}

def distance_profile(video, audio, vshift=15):
    if type(vshift) is not int or not 1 <= vshift <= 50:
        raise ValueError("SyncNet shift requires an integer from 1 to 50")
    for values in (video, audio):
        values = np.asarray(values)
        if values.ndim != 2 or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values) or not np.isfinite(values).all():
            raise ValueError("SyncNet embeddings must be finite real matrices")
    video, audio = torch.as_tensor(video, dtype=torch.float32), torch.as_tensor(audio, dtype=torch.float32)
    if video.shape != audio.shape or video.shape[0] < 2 * vshift + 1 or video.shape[1] < 1:
        raise ValueError("SyncNet embeddings are unequal or too short for the search")
    padded = torch.nn.functional.pad(audio, (0, 0, vshift, vshift))
    distances = []
    for index in range(len(video)):
        distances.append(torch.nn.functional.pairwise_distance(video[index:index + 1].expand(2 * vshift + 1, -1), padded[index:index + 2 * vshift + 1]))
    return torch.stack(distances).mean(0).numpy().astype(np.float64)

def valid_runs(mask):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return zip(edges[::2], edges[1::2])

class LipSyncScorer:
    def __init__(self, weights=DEFAULT_WEIGHTS, device="auto", batch_size=8, vshift=15, face_box=None, face_cropped=False, minimum_confidence=3.0, tolerance=0.08):
        if type(batch_size) is not int or not 1 <= batch_size <= 64 or type(vshift) is not int or not 1 <= vshift <= 50:
            raise ValueError("Invalid SyncNet batch size or search shift")
        if not np.isfinite([minimum_confidence, tolerance]).all() or minimum_confidence < 0 or tolerance < 0 or tolerance > vshift / 25:
            raise ValueError("SyncNet confidence and tolerance must be finite and nonnegative")
        self.faces = FaceReader(face_box, face_cropped)
        self.weights = relative_path(weights, True)
        self.model = load_expert(weights, device)
        self.model_id = checkpoint_identity(weights)
        self.device = next(self.model.parameters()).device
        self.batch_size, self.vshift = batch_size, vshift
        self.minimum_confidence, self.tolerance = minimum_confidence, tolerance

    def embeddings(self, frames, waveform):
        spectrum = mfcc_features(waveform)
        count = min(len(frames), len(waveform) // 640) - 5
        if count < 2 * self.vshift + 1:
            raise ValueError("Face track is too short for the configured offset search")
        video_features, audio_features = [], []
        with torch.inference_mode():
            for begin in range(0, count, self.batch_size):
                indices = range(begin, min(count, begin + self.batch_size))
                video = np.stack([frames[index:index + 5].transpose(3, 0, 1, 2) for index in indices]).astype(np.float32)
                audio = np.stack([spectrum[:, index * 4:index * 4 + 20] for index in indices])[:, None]
                video_features.append(self.model.forward_lip(torch.from_numpy(video).to(self.device)).cpu().numpy())
                audio_features.append(self.model.forward_aud(torch.from_numpy(audio).to(self.device)).cpu().numpy())
        return np.concatenate(video_features), np.concatenate(audio_features)

    def score_window(self, video_path, audio_path, start, duration, audio_offset):
        config = replace(Config(), image_size=32)
        clip = read_media(video_path, config, audio_path, start, duration, audio_offset)
        origin = 0.0 if audio_path is not None else None if importlib.util.find_spec("av") is not None else probe_stream_origins(video_path)["video"]
        frames, times, valid = self.faces.read(video_path, start, duration, origin)
        begin, end = max(clip.start, times[0]), min(clip.end, times[-1] + np.median(np.diff(times)))
        grid = begin + np.arange(max(0, int(np.floor((end - begin) * 25 + 1e-6)))) / 25
        if len(grid) < 2 * self.vshift + 6:
            raise ValueError("Shared color and audio coverage is too short")
        right = np.searchsorted(times, grid).clip(0, len(times) - 1)
        left = np.maximum(right - 1, 0)
        nearest = np.where(np.abs(times[left] - grid) <= np.abs(times[right] - grid), left, right)
        mask = valid[nearest] & (np.abs(times[nearest] - grid) <= 0.04 + 1e-8)
        tracks = []
        for left, right in valid_runs(mask):
            if right - left < 2 * self.vshift + 6:
                continue
            first = int(round((grid[left] - clip.audio_start) * 16000))
            last = first + (right - left) * 640
            if first < 0 or last > len(clip.waveform):
                continue
            waveform = clip.waveform[first:last]
            if np.sqrt(np.mean(waveform.astype(np.float64) ** 2)) < config.minimum_rms:
                continue
            video, audio = self.embeddings(frames[nearest[left:right]], waveform)
            curve = distance_profile(video, audio, self.vshift)
            tracks.append({"segment_start": float(grid[left]), "segment_end": float(grid[left] + (right - left) / 25), "embedding_count": len(video), **profile_summary(curve, self.vshift, self.minimum_confidence, self.tolerance)})
        if not tracks:
            raise ValueError("No sufficiently long unambiguous face track with usable speech audio")
        return tracks

    def predict(self, video_path, audio_path=None, start=0.0, duration=None, audio_offset=0.0, window_seconds=5.0, output=None):
        if output is not None:
            relative_path(output)
            for path in (video_path, audio_path, getattr(self, "weights", None)):
                if path is not None and paths_match(relative_path(path), output):
                    raise ValueError("Lip-sync output cannot overwrite media or expert weights")
        if not np.isfinite([start, audio_offset, window_seconds]).all() or start < 0 or not 1.5 <= window_seconds <= 30:
            raise ValueError("Invalid lip-sync interval or window size")
        if window_seconds * 25 < 2 * self.vshift + 6:
            raise ValueError("Lip-sync window is too short for the search shift")
        if duration is not None and (not np.isfinite(duration) or duration <= 0):
            raise ValueError("Lip-sync duration must be finite and positive")
        total = probe_duration(video_path)
        end = total if duration is None else min(total, start + duration)
        if start >= end:
            raise ValueError("Lip-sync start is outside the video")
        tracks, failures = [], []
        for index in range(int(math.ceil((end - start) / window_seconds - 1e-9))):
            begin = start + index * window_seconds
            span = min(window_seconds, end - begin)
            try:
                tracks.extend(self.score_window(video_path, audio_path, begin, span, audio_offset))
            except (ValueError, OSError, RuntimeError) as error:
                failures.append({"segment_start": begin, "segment_end": begin + span, "reason": str(error)})
        report = {"metric": "SyncNet v2 LSE-C/LSE-D", "status": "scored" if tracks else "insufficient_evidence", "lse_c": None, "lse_d": None, "audio_lag_seconds": None, "model_id": self.model_id, "face_mode": self.faces.mode, "protocol": "mwsc_bounded_tracks_v1", "fps": 25, "sample_rate": 16000, "search_frames": self.vshift, "minimum_confidence": self.minimum_confidence, "tolerance_seconds": self.tolerance, "tracks": tracks, "failures": failures, "scored_tracks": len(tracks), "aggregation": "embedding_count_weighted_distance_curve", "warnings": ["LSE-C is higher better; LSE-D is lower better; neither is a fake probability", "High confidence may describe a delayed track; inspect audio_lag_seconds", "Haar or manual crops differ from the official S3FD benchmark preprocessing"]}
        coverage = sum(item["segment_end"] - item["segment_start"] for item in tracks) / (end - start)
        report["coverage"] = float(min(1, coverage))
        if tracks:
            count = sum(item["embedding_count"] for item in tracks)
            curve = sum(np.asarray(item["distance_curve"]) * item["embedding_count"] for item in tracks) / count
            report.update(profile_summary(curve, self.vshift, self.minimum_confidence, self.tolerance))
            report["embedding_count"] = count
        if output is not None:
            atomic_json(output, report)
        return report

def add_lipsync(result, video_path, audio_path=None, **options):
    options = dict(options)
    media = {name: options.pop(name) for name in ("start", "duration", "audio_offset", "window_seconds") if name in options}
    try:
        result["lip_sync"] = LipSyncScorer(**options).predict(video_path, audio_path, **media)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        result["lip_sync"] = {"metric": "SyncNet v2 LSE-C/LSE-D", "status": "unavailable", "reason": str(error), "lse_c": None, "lse_d": None, "audio_lag_seconds": None}
    return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
