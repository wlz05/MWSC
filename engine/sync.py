# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import numpy as np
from itertools import chain
from core.align import align_features, AlignmentError
from engine.detect import Detector, score_features
from dataset.manifest import select_rows
from core.media import MediaClip, read_media
from support.metrics import binary_metrics
from dataset.split import audit_splits
from support.storage import atomic_json, protect_output
from core.wavelet import first_component, InsufficientEvidence

def estimate_offset(video, audio, video_times, audio_times, config, maximum_offset=0.5):
    if not np.isfinite(maximum_offset) or maximum_offset <= 0:
        raise ValueError("Maximum offset must be positive and finite")
    video, audio, grid, alignment = align_features(video, audio, video_times, audio_times, config)
    left, _ = first_component(video, config.minimum_feature_variance)
    right, _ = first_component(audio, config.minimum_feature_variance)
    maximum_frames = min(int(np.floor(maximum_offset * config.video_fps)), len(grid) - config.minimum_samples)
    minimum_overlap = max(config.minimum_samples, len(grid) // 2)
    candidates = []
    for lag in range(-maximum_frames, maximum_frames + 1):
        if lag > 0:
            first, second = left[:-lag], right[lag:]
        elif lag < 0:
            first, second = left[-lag:], right[:lag]
        else:
            first, second = left, right
        if len(first) < minimum_overlap:
            continue
        first = first - first.mean()
        second = second - second.mean()
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator <= 1e-12:
            continue
        correlation = float(np.clip(abs(np.dot(first, second)) / denominator, 0, 1))
        candidates.append({"lag_frames": lag, "offset_seconds": lag / config.video_fps, "correlation": correlation, "overlap_samples": len(first)})
    if not candidates:
        raise InsufficientEvidence("No valid lag candidate")
    ranked = sorted(candidates, key=lambda item: (-item["correlation"], abs(item["lag_frames"])))
    best = ranked[0]
    competitors = [item["correlation"] for item in ranked if abs(item["lag_frames"] - best["lag_frames"]) > 1]
    margin = best["correlation"] - max(competitors, default=0)
    return {"estimated_offset_seconds": best["offset_seconds"], "estimated_offset_frames": best["lag_frames"], "peak_correlation": best["correlation"], "peak_margin": float(margin), "ambiguous_peak": bool(margin < 0.02), "at_search_boundary": abs(best["lag_frames"]) == maximum_frames, "offset_resolution_seconds": 1 / config.video_fps, "convention": "Positive means audio content lags video content", "method": "PCA temporal correlation; an engineering diagnostic", "alignment": alignment, "candidates": candidates}

def shifted_clip(clip, offset_seconds):
    if not np.isfinite(offset_seconds):
        raise ValueError("Audio shift must be finite")
    shift = int(round(offset_seconds * clip.sample_rate))
    if abs(shift) >= len(clip.waveform):
        raise InsufficientEvidence("Audio shift removes the entire overlap")
    if shift > 0:
        waveform = clip.waveform[:-shift].copy()
        audio_start = clip.audio_start + shift / clip.sample_rate
    elif shift < 0:
        waveform = clip.waveform[-shift:].copy()
        audio_start = clip.audio_start
    else:
        waveform = clip.waveform.copy()
        audio_start = clip.audio_start
    audio_end = audio_start + len(waveform) / clip.sample_rate
    mask = (clip.video_times >= audio_start - 1e-8) & (clip.video_times < audio_end - 1e-8)
    if mask.sum() < 2:
        raise InsufficientEvidence("Audio shift leaves too few video frames")
    warnings = list(clip.warnings) + ["Controlled audio content shift: %.6f seconds" % (shift / clip.sample_rate)]
    return MediaClip(clip.frames[mask].copy(), clip.video_times[mask].copy(), waveform, audio_start, clip.sample_rate, warnings)

def sync_measurement(detector, clip, tolerance_seconds=0.05, maximum_offset=0.5):
    if not np.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("Synchronization tolerance must be nonnegative")
    features = detector.extractor.extract(clip)
    offset = estimate_offset(*features, detector.config, maximum_offset)
    patent = score_features(*features, detector.config, detector.threshold, False)
    asynchronous = abs(offset["estimated_offset_seconds"]) > tolerance_seconds
    return {**offset, "sync_decision": "out_of_sync" if asynchronous else "synchronized", "predicted_sync_label": int(asynchronous), "synchronization_score": patent["synchronization_score"], "patent_decision": patent["decision"], "threshold": patent["threshold_before"], "valid_duration": clip.end - clip.start}

def evaluate_sync(rows, checkpoint, split="test", device="auto", config_override=None, offsets=None, tolerance_seconds=0.05, maximum_offset=0.5, limit=None, output=None):
    if split not in ("val", "test"):
        raise ValueError("Synchronization evaluation requires val or test")
    if not np.isfinite(tolerance_seconds) or tolerance_seconds < 0 or not np.isfinite(maximum_offset) or maximum_offset <= 0:
        raise ValueError("Synchronization bounds must be finite and valid")
    if offsets is not None:
        try:
            # Materialize once so generators survive validation and evaluation.
            offsets = list(offsets)
        except TypeError as error:
            raise ValueError("Controlled offsets must be a sequence of real numbers") from error
        if not offsets or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or abs(value) > maximum_offset for value in offsets):
            raise ValueError("Controlled offsets must be nonempty and inside the search range")
        offsets = [float(value) for value in offsets]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Synchronization evaluation limit must be a positive integer")
    audit = audit_splits(rows)
    protect_output(output, chain((checkpoint,), (row.get(key) for row in rows for key in ("video", "audio"))))
    selected = select_rows(rows, split, real_only=offsets is not None)
    if limit is not None:
        if limit < 1:
            raise ValueError("Synchronization evaluation limit must be positive")
        selected = selected[:limit]
    detector = Detector(checkpoint, device, config_override=config_override)
    outcomes = []
    labels, scores, decisions, patent_scores, patent_decisions, errors = [], [], [], [], [], []
    for row in selected:
        applied = [None] if offsets is None else offsets
        if offsets is not None and row.get("sync_label", "") != 0:
            outcomes.extend({"video": row["video"], "applied_offset_seconds": value, "decision": "insufficient_evidence", "reason": "Controlled shifts require an explicitly synchronized baseline"} for value in applied)
            continue
        try:
            baseline = read_media(row["video"], detector.config, row["audio"] or None, duration=detector.config.segment_seconds, audio_offset=row["audio_offset"])
            if baseline.end - baseline.start < detector.config.segment_seconds * 0.8:
                raise ValueError("Baseline has insufficient shared audio-video coverage")
        except (OSError, ValueError, RuntimeError) as error:
            outcomes.extend({"video": row["video"], "applied_offset_seconds": value, "decision": "insufficient_evidence", "reason": str(error)} for value in applied)
            continue
        for value in applied:
            expected = row.get("offset_seconds", "")
            label = row.get("sync_label", "")
            if value is not None:
                shift = round(value * baseline.sample_rate) / baseline.sample_rate
                if expected != "":
                    expected = float(expected) + shift
                    label = int(abs(expected) > tolerance_seconds)
                else:
                    # Unknown baseline lag is not a zero-lag annotation.
                    label = 0 if shift == 0 else 1 if abs(shift) > 2 * tolerance_seconds else ""
            try:
                clip = shifted_clip(baseline, value) if value is not None else baseline
                result = sync_measurement(detector, clip, tolerance_seconds, maximum_offset)
                result.update({"video": row["video"], "deepfake_label": int(row["label"]), "true_sync_label": label, "expected_offset_seconds": expected, "applied_offset_seconds": value, "decision": "scored"})
                if expected != "":
                    error = abs(result["estimated_offset_seconds"] - expected)
                    result["offset_absolute_error_seconds"] = error
                    errors.append(error)
                if label != "":
                    labels.append(label)
                    scores.append(abs(result["estimated_offset_seconds"]))
                    decisions.append(result["predicted_sync_label"])
                    patent_scores.append(-result["synchronization_score"])
                    patent_decisions.append(int(result["patent_decision"] == "fake"))
            except (InsufficientEvidence, AlignmentError, ValueError, RuntimeError) as error:
                result = {"video": row["video"], "applied_offset_seconds": value, "decision": "insufficient_evidence", "reason": str(error)}
            outcomes.append(result)
    scored = sum(item["decision"] == "scored" for item in outcomes)
    report = {"project": "MWSC", "split": split, "controlled_offsets_seconds": offsets, "tolerance_seconds": tolerance_seconds, "maximum_offset_seconds": maximum_offset, "attempted_pairs": len(outcomes), "scored_pairs": scored, "coverage": scored / len(outcomes) if outcomes else 0, "offset_metrics": {"annotated_pairs": len(errors), "mean_absolute_error_seconds": float(np.mean(errors)) if errors else None, "within_tolerance_fraction": float(np.mean(np.asarray(errors) <= tolerance_seconds)) if errors else None}, "offset_sync_metrics": binary_metrics(labels, scores, decisions) if labels else None, "patent_sync_metrics": binary_metrics(labels, patent_scores, patent_decisions) if labels else None, "split_audit": audit, "outcomes": outcomes, "notes": ["Synchronization annotations are separate from authenticity labels", "Controlled shifts retain original authenticity labels", "Offset correlation supplements the patent formula and is not SyncNet", "Constant phase shifts can preserve patent coherence", "Periodic signals and weak neural features can give ambiguous offset peaks"]}
    if output:
        atomic_json(output, report)
    return report

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
