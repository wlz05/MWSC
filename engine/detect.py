# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import hashlib
import json
import zipfile
import numpy as np
from core.align import align_features, AlignmentError
from support.storage import numpy_save, relative_path
from core.media import iter_windows
from core.threshold import DynamicThreshold
from core.wavelet import synchronization_score, InsufficientEvidence

def checkpoint_identity(path):
    digest = hashlib.sha256()
    with relative_path(path, True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()

def score_features(video, audio, video_times, audio_times, config, threshold=None, update=True):
    aligned_video, aligned_audio, grid, diagnostics = align_features(video, audio, video_times, audio_times, config)
    result = synchronization_score(aligned_video, aligned_audio, config)
    result["alignment"] = diagnostics
    result["feature_interval"] = [float(grid[0]), float(grid[-1])]
    if threshold is not None:
        result.update(threshold.decide(result["synchronization_score"], update))
    return result

def save_features(path, features, config, model_id):
    video, audio, video_times, audio_times = features
    numpy_save(path, video=video, audio=audio, video_times=video_times, audio_times=audio_times, config=np.asarray(json.dumps(config.to_dict())), version=np.asarray(1), model_id=np.asarray(model_id))

def load_features(path):
    try:
        archive = np.load(relative_path(path, True), allow_pickle=False)
    except (EOFError, zipfile.BadZipFile) as error:
        raise ValueError("Feature archive cannot be decoded") from error
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Feature input must be an NPZ archive")
    with archive:
        required = {"video", "audio", "video_times", "audio_times", "config", "version", "model_id"}
        if not required.issubset(archive.files):
            raise ValueError("Unsupported feature archive")
        version = archive["version"]
        if version.ndim != 0 or not np.issubdtype(version.dtype, np.integer) or int(version) != 1:
            raise ValueError("Unsupported feature archive")
        from config import Config
        config = Config.from_dict(json.loads(str(archive["config"])))
        features = tuple(np.asarray(archive[key]).copy() for key in ("video", "audio", "video_times", "audio_times"))
        model_id = str(archive["model_id"])
    return features, config, model_id

class Detector:
    def __init__(self, checkpoint, device="auto", stream_id="default", config_override=None):
        from model.weights import load_encoders
        from model.extract import NeuralExtractor
        model, config, self.metadata = load_encoders(checkpoint, device)
        if config_override is not None:
            before, after = config.to_dict(), config_override.to_dict()
            permitted = {"threshold", "history_size", "ema_alpha", "tolerance_k", "weight_mode", "frequency_count", "frequency_min", "frequency_max", "gaussian_mean", "gaussian_std", "morlet_omega"}
            changed = {key for key in before if before[key] != after[key]}
            if changed - permitted:
                raise ValueError("Inference preprocessing must match encoder training")
            config = config_override.validate()
        self.config = config
        self.model_id = checkpoint_identity(checkpoint)
        self.extractor = NeuralExtractor(model, config)
        self.threshold = DynamicThreshold(config, stream_id, self.model_id)

    def process_segment(self, clip, update=True):
        try:
            clip.validate()
            start, end = float(clip.start), float(clip.end)
            if not np.isfinite([start, end]).all() or end <= start:
                raise ValueError("Media has no finite shared interval")
        except ValueError as error:
            return {"decision": "insufficient_evidence", "reason": str(error), "updated": False, "segment_start": None, "segment_end": None}
        try:
            features = self.extractor.extract(clip)
            result = score_features(*features, self.config, self.threshold, update)
            result["warnings"].extend(clip.warnings)
        except (InsufficientEvidence, AlignmentError) as error:
            result = {"decision": "insufficient_evidence", "reason": str(error), "updated": False}
        result["segment_start"] = start
        result["segment_end"] = end
        return result

    def predict(self, video_path, audio_path=None, start=0.0, duration=None, audio_offset=0.0, update=True):
        results = []
        for index, begin, end, segment, error in iter_windows(video_path, self.config, audio_path, start, duration, audio_offset):
            result = self.process_segment(segment, update) if error is None else {"decision": "insufficient_evidence", "reason": error, "updated": False}
            result["segment_index"] = index
            result["segment_start"] = begin
            result["segment_end"] = end
            results.append(result)
        if not results:
            return {"decision": "insufficient_evidence", "reason": "No complete segment in the shared interval", "segments": [], "scores": {"mwsc_mean": None, "mwsc_min": None, "mwsc_normalized_mean": None}}
        valid = [item for item in results if "synchronization_score" in item]
        if any(item["decision"] == "fake" for item in valid):
            decision = "fake"
        elif len(valid) == len(results):
            decision = "real"
        else:
            decision = "insufficient_evidence"
        scores = {"mwsc_mean": float(np.mean([item["synchronization_score"] for item in valid])) if valid else None, "mwsc_min": min((item["synchronization_score"] for item in valid), default=None), "mwsc_normalized_mean": float(np.mean([item["synchronization_score"] / item["score_upper_bound"] for item in valid])) if valid else None}
        return {"project": "MWSC", "decision": decision, "aggregation": "any_fake_segment", "model_id": self.model_id, "scores": scores, "segments": results, "valid_segments": len(valid), "total_segments": len(results), "window_coverage": len(valid) / len(results), "bounded_window_decoding": True}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
