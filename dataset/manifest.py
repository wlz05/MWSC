# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from collections import defaultdict
import json
import math
from pathlib import Path
from numbers import Integral
from support.storage import read_csv, relative_path, write_csv

COLUMNS = ("dataset", "video", "audio", "label", "group", "source", "split", "duration", "audio_offset", "fake_segments", "category", "audio_segments", "video_segments", "sync_label", "offset_seconds", "identity_namespace", "source_namespace")
SPLITS = ("train", "val", "test")

def canonical_split(value):
    value = str(value).strip().lower()
    return {"dev": "val", "valid": "val", "validation": "val", "training": "train", "testing": "test"}.get(value, value)

def parse_segments(value):
    if value in (None, "", "[]"):
        return []
    segments = json.loads(value) if isinstance(value, str) else value
    if not isinstance(segments, (list, tuple)):
        raise ValueError("Fake intervals must be an array")
    result = []
    for segment in segments:
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            raise ValueError("Each fake interval requires start and end seconds")
        try:
            start, end = map(float, segment)
        except (TypeError, ValueError) as error:
            raise ValueError("Fake interval endpoints must be numbers") from error
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("Invalid fake interval")
        result.append([start, end])
    return result

def binary_label(value, name):
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value.strip())
    if isinstance(value, Integral) and not isinstance(value, bool) and value in (0, 1):
        return int(value)
    raise ValueError(name + " must be the integer 0 or 1")

def validate_rows(rows, check_files=False, require_split=False):
    if not rows:
        raise ValueError("Manifest is empty")
    seen = set()
    result = []
    for index, original in enumerate(rows):
        if not isinstance(original, dict):
            raise ValueError("Manifest rows must be dictionaries")
        row = {key: "" if original.get(key) is None else original.get(key, "") for key in COLUMNS}
        for required in ("dataset", "video", "label", "group", "source"):
            if str(row[required]).strip() == "":
                raise ValueError("Missing manifest field at row %d: %s" % (index, required))
        row["video"] = str(relative_path(row["video"], check_files)).replace("\\", "/")
        if row["audio"]:
            row["audio"] = str(relative_path(row["audio"], check_files)).replace("\\", "/")
        row["label"] = binary_label(row["label"], "Authenticity label")
        for key in ("dataset", "group", "source", "category", "identity_namespace", "source_namespace"):
            row[key] = str(row[key]).strip()
        if not any(identity.strip() for identity in row["group"].split(";")):
            raise ValueError("Identity group cannot contain only separators")
        row["split"] = canonical_split(row["split"])
        if row["split"] not in (*SPLITS, "") or require_split and not row["split"]:
            raise ValueError("Invalid or missing manifest split")
        row["duration"] = float(row["duration"]) if row["duration"] != "" else ""
        if row["duration"] != "" and (not math.isfinite(row["duration"]) or row["duration"] <= 0):
            raise ValueError("Invalid manifest duration")
        row["audio_offset"] = float(row["audio_offset"] or 0)
        if not math.isfinite(row["audio_offset"]):
            raise ValueError("Invalid audio offset")
        segments = parse_segments(row["fake_segments"])
        audio_segments = parse_segments(row["audio_segments"])
        video_segments = parse_segments(row["video_segments"])
        all_segments = segments + audio_segments + video_segments
        if row["label"] == 0 and all_segments:
            raise ValueError("Real samples cannot have fake intervals")
        if row["duration"] != "" and any(end > row["duration"] + 0.1 for _, end in all_segments):
            raise ValueError("Fake interval exceeds video duration")
        segments = []
        for start, end in sorted(all_segments):
            if segments and start <= segments[-1][1]:
                segments[-1][1] = max(segments[-1][1], end)
            else:
                segments.append([start, end])
        row["fake_segments"] = json.dumps(segments, separators=(",", ":"))
        row["audio_segments"] = json.dumps(audio_segments, separators=(",", ":"))
        row["video_segments"] = json.dumps(video_segments, separators=(",", ":"))
        row["sync_label"] = binary_label(row["sync_label"], "Synchronization label") if row["sync_label"] != "" else ""
        row["offset_seconds"] = float(row["offset_seconds"]) if row["offset_seconds"] != "" else ""
        if row["offset_seconds"] != "" and not math.isfinite(row["offset_seconds"]):
            raise ValueError("Invalid synchronization offset")
        key = str(Path(row["video"]).resolve()).casefold()
        if key in seen:
            raise ValueError("Duplicate video in manifest: " + row["video"])
        seen.add(key)
        result.append(row)
    return result

def read_manifest(path, check_files=False, require_split=False):
    return validate_rows(read_csv(path), check_files, require_split)

def save_manifest(path, rows):
    write_csv(path, validate_rows(rows), COLUMNS)

def select_rows(rows, split, real_only=False):
    split = canonical_split(split)
    selected = [row for row in rows if row["split"] == split and (not real_only or row["label"] == 0 and row.get("sync_label", "") in ("", 0))]
    if not selected:
        raise ValueError("No eligible rows in split: " + split)
    return selected

def summary(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[(row["dataset"], row["split"] or "unsplit", int(row["label"]))] += 1
    return {"rows": len(rows), "groups": len({row["group"] for row in rows}), "counts": [{"dataset": key[0], "split": key[1], "label": key[2], "count": count} for key, count in sorted(counts.items())]}

def window_label(row, start, end):
    intervals = parse_segments(row["fake_segments"])
    if not intervals:
        return int(row["label"])
    return int(any(max(start, left) < min(end, right) for left, right in intervals))

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
