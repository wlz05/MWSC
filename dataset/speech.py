# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from pathlib import Path
import numpy as np
from dataset.adapter import identity_from_path, joined_path
from dataset.manifest import canonical_split, validate_rows
from core.media import read_media
from support.storage import relative_path

DATASETS = ("lrs2", "lrs3", "voxceleb2")

def import_speech(root, dataset, filelist=None, subset="", audio_root=None):
    if dataset not in DATASETS:
        raise ValueError("Unknown speech dataset")
    subset = canonical_split(subset)
    if subset not in ("", "train", "val", "test"):
        raise ValueError("Choose train, val, test, or an unsplit speech import")
    base = relative_path(root, True)
    if filelist:
        entries = []
        for line in relative_path(filelist, True).read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            entry = relative_path(line.split()[0])
            entries.append(entry.with_suffix(".mp4") if not entry.suffix else entry)
    else:
        entries = [path.relative_to(base) for path in sorted(base.rglob("*")) if path.is_file() and path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]
    rows = []
    for entry in entries:
        if entry.suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv"):
            raise ValueError("Speech file lists must identify video clips")
        video = joined_path(base, entry)
        relative_path(video, True)
        identity = identity_from_path(entry)
        parent = entry.parent.as_posix()
        source = parent if parent != "." else entry.stem
        group = identity or "source:" + source
        audio = joined_path(audio_root, entry.with_suffix(".wav")) if audio_root else ""
        if audio:
            relative_path(audio, True)
        rows.append({"dataset": dataset, "video": video, "audio": audio, "label": 0, "group": group, "source": source, "split": subset, "duration": "", "audio_offset": 0, "fake_segments": "[]", "category": "speech_real", "sync_label": "", "offset_seconds": ""})
    return validate_rows(rows)

def audit_media(rows, config, limit=None):
    if limit is not None and limit < 1:
        raise ValueError("Audit limit must be positive")
    selected = rows if limit is None else rows[:limit]
    outcomes = []
    for row in selected:
        try:
            clip = read_media(row["video"], config, row["audio"] or None, duration=config.segment_seconds, audio_offset=row["audio_offset"])
            native_rate = 1 / float(np.median(np.diff(clip.video_times)))
            rms = float(np.sqrt(np.mean(clip.waveform.astype(np.float64) ** 2)))
            passed = native_rate > 2 * config.frequency_max and rms >= config.minimum_rms and clip.end - clip.start >= config.segment_seconds * 0.8
            outcomes.append({"video": row["video"], "passed": bool(passed), "native_video_rate": native_rate, "audio_rms": rms, "shared_duration": clip.end - clip.start, "warnings": clip.warnings})
        except (OSError, ValueError, RuntimeError) as error:
            outcomes.append({"video": row["video"], "passed": False, "reason": str(error)})
    return {"checked": len(selected), "passed": sum(item["passed"] for item in outcomes), "failed": sum(not item["passed"] for item in outcomes), "samples": outcomes}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
