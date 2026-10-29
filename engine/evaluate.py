# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import math
from itertools import chain
import numpy as np
from support.storage import atomic_json, protect_output
from core.media import read_media, probe_duration
from engine.detect import Detector
from core.threshold import DynamicThreshold
from dataset.manifest import select_rows, parse_segments, window_label
from dataset.split import audit_splits
from dataset.provenance import verify_holdout
from support.metrics import binary_metrics, localization_metrics

def evaluate_row(detector, row, adaptive=False):
    config = detector.config
    try:
        duration = probe_duration(row["video"])
        if row["duration"] != "" and (not np.isfinite(float(row["duration"])) or abs(float(row["duration"]) - duration) > 0.1):
            raise ValueError("Manifest duration differs from decoded video duration")
    except (OSError, ValueError, RuntimeError) as error:
        return {"video": row["video"], "decision": "insufficient_evidence", "reason": str(error), "segments": [], "valid_segments": 0, "total_segments": 0, "minimum_sync_score": None}
    step = config.segment_seconds * (1 - config.overlap)
    count = int(np.floor((duration - config.segment_seconds) / step + 1e-8)) + 1
    if count < 1:
        return {"video": row["video"], "decision": "insufficient_evidence", "reason": "Video is shorter than one segment", "segments": []}
    # Reset adaptive history before evaluating each video.
    detector.threshold = DynamicThreshold(config, row["video"], detector.model_id)
    segments = []
    for index in range(count):
        start = index * step
        try:
            clip = read_media(row["video"], config, row["audio"] or None, start, config.segment_seconds, row["audio_offset"])
            if clip.end - clip.start < config.segment_seconds * 0.8:
                raise ValueError("Window has insufficient shared audio-video coverage")
            result = detector.process_segment(clip, adaptive)
        except (OSError, ValueError, RuntimeError) as error:
            result = {"decision": "insufficient_evidence", "reason": str(error), "updated": False}
        result["segment_start"] = start
        result["segment_end"] = start + config.segment_seconds
        result["true_label"] = window_label(row, start, start + config.segment_seconds)
        segments.append(result)
    valid = [item for item in segments if "synchronization_score" in item]
    if any(item["decision"] == "fake" for item in valid):
        decision = "fake"
    elif len(valid) == len(segments):
        decision = "real"
    else:
        decision = "insufficient_evidence"
    return {"video": row["video"], "dataset": row["dataset"], "category": row["category"], "true_label": int(row["label"]), "decision": decision, "segments": segments, "valid_segments": len(valid), "total_segments": len(segments), "minimum_sync_score": min((item["synchronization_score"] for item in valid), default=None)}

def evaluate(rows, checkpoint, split="test", device="auto", config_override=None, adaptive=False, limit=None, output=None):
    if split not in ("val", "test"):
        raise ValueError("Evaluation requires a validation or test split")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Evaluation limit must be a positive integer")
    audit = audit_splits(rows)
    # Protect model and media inputs for direct Python callers too.
    protect_output(output, chain((checkpoint,), (row.get(key) for row in rows for key in ("video", "audio"))))
    selected = select_rows(rows, split)
    if limit is not None:
        if limit < 1:
            raise ValueError("Evaluation limit must be positive")
        selected = selected[:limit]
    detector = Detector(checkpoint, device, config_override=config_override)
    verify_holdout(rows, detector.metadata, split)
    predictions = [evaluate_row(detector, row, adaptive) for row in selected]
    labels, scores, decisions = [], [], []
    window_labels, window_scores, window_predictions = [], [], []
    categories = {}
    localized = []
    truth = {}
    for row, prediction in zip(selected, predictions):
        truth[row["video"]] = parse_segments(row["fake_segments"])
        if prediction["decision"] in ("real", "fake"):
            labels.append(row["label"])
            # Higher ROC scores must indicate stronger evidence of forgery.
            scores.append(-prediction["minimum_sync_score"])
            decisions.append(int(prediction["decision"] == "fake"))
            group = categories.setdefault(row["category"] or "unspecified", {"total": 0, "correct": 0})
            group["total"] += 1
            group["correct"] += int(decisions[-1] == row["label"])
        for segment in prediction["segments"]:
            if "synchronization_score" not in segment:
                continue
            window_labels.append(segment["true_label"])
            window_scores.append(-segment["synchronization_score"])
            window_predictions.append(segment["predicted_label"])
            if segment["decision"] == "fake":
                localized.append((row["video"], segment["segment_start"], segment["segment_end"], -segment["synchronization_score"]))
    report = {"split": split, "adaptive_within_video": adaptive, "state_reset_between_videos": True, "requested_videos": len(selected), "scored_videos": len(labels), "video_coverage": len(labels) / len(selected), "clip_metrics": binary_metrics(labels, scores, decisions) if labels else None, "window_metrics": binary_metrics(window_labels, window_scores, window_predictions) if window_labels else None, "categories": categories, "window_proposal_localization": localization_metrics(localized, truth), "split_audit": audit, "predictions": predictions, "notes": ["Fake-positive ROC uses negative synchronization score", "Overlapping windows are correlated observations", "Window proposal AP is not the official dataset benchmark protocol", "Report failed samples and coverage with every metric"]}
    if output:
        atomic_json(output, report)
    return report

def calibrate(rows, checkpoint, false_positive_rate=0.05, device="auto", output=None):
    if isinstance(false_positive_rate, bool) or not isinstance(false_positive_rate, (int, float)) or not math.isfinite(false_positive_rate) or not 0 <= false_positive_rate < 1:
        raise ValueError("False positive rate must lie in [0, 1)")
    audit_splits(rows)
    protect_output(output, chain((checkpoint,), (row.get(key) for row in rows for key in ("video", "audio"))))
    # Calibrate the threshold from held-out real videos only.
    real_validation = select_rows(rows, "val", real_only=True)
    detector = Detector(checkpoint, device)
    verify_holdout(rows, detector.metadata, "val")
    scores, failures = [], []
    for row in real_validation:
        prediction = evaluate_row(detector, row, False)
        if prediction.get("valid_segments") == prediction.get("total_segments") and prediction.get("minimum_sync_score") is not None:
            scores.append(prediction["minimum_sync_score"])
        else:
            failures.append(row["video"])
    if len(scores) < 2:
        raise ValueError("Calibration requires at least two fully scored real validation videos")
    threshold = float(np.quantile(scores, false_positive_rate, method="lower"))
    values = detector.config.to_dict()
    values["threshold"] = threshold
    result = {"config": values, "model_id": detector.model_id, "method": "lower_quantile_of_real_validation_video_minimum_scores", "validation_scores": scores, "validation_rows": real_validation, "failed_videos": failures, "target_false_positive_rate": false_positive_rate, "empirical_false_positive_rate": float(np.mean(np.asarray(scores) < threshold)), "notes": ["Calibration is an engineering extension to the preferred patent threshold", "Test labels are never used for threshold selection"]}
    if output:
        atomic_json(output, result)
    return result

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
