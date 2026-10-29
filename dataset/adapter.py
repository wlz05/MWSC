# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import json
from pathlib import Path
import re
from support.storage import read_json, read_csv, relative_path
from dataset.manifest import canonical_split, validate_rows

def identity_from_path(value):
    identities = sorted(set(re.findall(r"(?<![a-zA-Z0-9])id\d+", str(value))))
    return ";".join(identities)

def joined_path(root, value):
    child = relative_path(str(value).replace("\\", "/"))
    # Keep metadata entries inside the selected dataset root.
    if ".." in child.parts:
        raise ValueError("Dataset metadata path cannot escape its root")
    return str(relative_path(root) / child).replace("\\", "/")

def metadata_list(path):
    data = read_json(path)
    if isinstance(data, dict):
        for key in ("metadata", "data", "videos"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("Dataset metadata requires a list of sample dictionaries")
    return data

def import_avdeepfake(root, metadata):
    rows = []
    allowed = {"real", "visual_modified", "audio_modified", "both_modified"}
    for item in metadata_list(metadata):
        category = item["modify_type"]
        if category not in allowed:
            raise ValueError("Unknown AV-Deepfake1M modification type")
        source = item.get("original") or item["file"]
        identity = str(item.get("speaker_id") or item.get("subject_id") or identity_from_path(source))
        subset = canonical_split(item.get("split", ""))
        filename = str(item["file"]).replace('\\', '/')
        # The released SDK resolves media under its subset directory.
        if subset and filename.split('/')[0] not in ('train', 'val', 'test'):
            filename = subset + '/' + filename
        duration = item.get("duration", "")
        if duration == "" and "video_frames" in item:
            duration = float(item['video_frames']) / 25.0
        row = {"dataset": "avdeepfake1m", "video": joined_path(root, filename), "audio": "", "label": int(category != "real"), "group": identity or "source:" + source, "source": source, "split": subset, "duration": duration, "audio_offset": 0, "fake_segments": json.dumps(item.get("fake_segments", [])), "category": category, "audio_segments": json.dumps(item.get("audio_fake_segments", [])), "video_segments": json.dumps(item.get("visual_fake_segments", [])), "sync_label": item.get("sync_label", ""), "offset_seconds": item.get("offset_seconds", "")}
        rows.append(row)
    return validate_rows(rows)

def import_lavdf(root, metadata):
    rows = []
    for item in metadata_list(metadata):
        source = item.get("original") or item["file"]
        identity = str(item.get("speaker_id") or item.get("subject_id") or identity_from_path(source))
        intervals = item.get("fake_periods", [])
        fake = int(item.get("n_fakes", len(intervals))) > 0
        if not fake and (item.get("modify_audio") or item.get("modify_video")):
            raise ValueError("LAV-DF labels conflict with modification flags")
        category = "both_modified" if item.get("modify_audio") and item.get("modify_video") else "audio_modified" if item.get("modify_audio") else "visual_modified" if item.get("modify_video") else "real"
        rows.append({"dataset": "lavdf", "video": joined_path(root, item["file"]), "audio": "", "label": int(fake), "group": identity or "source:" + source, "source": source, "split": canonical_split(item.get("split", "")), "duration": item.get("duration", ""), "audio_offset": 0, "fake_segments": json.dumps(intervals), "category": category, "audio_segments": json.dumps(intervals if item.get("modify_audio") else []), "video_segments": json.dumps(intervals if item.get("modify_video") else []), "sync_label": item.get("sync_label", ""), "offset_seconds": item.get("offset_seconds", "")})
    return validate_rows(rows)

def import_fakeavceleb(root):
    root_path = relative_path(root, True)
    categories = {"realvideo-realaudio": (0, "real"), "realvideo-fakeaudio": (1, "audio_modified"), "fakevideo-realaudio": (1, "visual_modified"), "fakevideo-fakeaudio": (1, "both_modified")}
    rows = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv"):
            continue
        relative = path.relative_to(root_path)
        names = {part.lower() for part in relative.parts}
        matched = [key for key in categories if key in names]
        if len(matched) != 1:
            continue
        identity = identity_from_path(relative)
        if not identity:
            raise ValueError("FakeAVCeleb path needs a subject id; use a custom CSV manifest")
        label, category = categories[matched[0]]
        source = identity.split(";")[0] + "/" + path.stem.split("_")[0]
        rows.append({"dataset": "fakeavceleb", "video": str(path).replace("\\", "/"), "audio": "", "label": label, "group": identity, "source": source, "split": "", "duration": "", "audio_offset": 0, "fake_segments": "[]", "category": category})
    return validate_rows(rows)

def import_dfdc(root, metadata):
    data = read_json(metadata)
    if not isinstance(data, dict):
        raise ValueError("DFDC metadata requires a filename dictionary")
    rows = []
    for filename, item in sorted(data.items()):
        label = str(item["label"]).upper()
        if label not in ("REAL", "FAKE"):
            raise ValueError("Unknown DFDC label")
        source = item.get("original") or filename
        identity = str(item.get("actor_id") or item.get("subject_id") or "source:" + source)
        rows.append({"dataset": "dfdc", "video": joined_path(root, filename), "audio": "", "label": int(label == "FAKE"), "group": identity, "source": source, "split": canonical_split(item.get("split", "")), "duration": item.get("duration", ""), "audio_offset": 0, "fake_segments": "[]", "category": "visual_modified" if label == "FAKE" else "real"})
    return validate_rows(rows)

def apply_identity_map(rows, path):
    identities = {}
    for item in read_csv(path):
        source = item["source"]
        group = item["group"]
        if not group or source in identities and identities[source] != group:
            raise ValueError("Invalid or conflicting identity map")
        identities[source] = group
    missing = {row["source"] for row in rows} - set(identities)
    if missing:
        raise ValueError("Identity map lacks %d original sources" % len(missing))
    return [{**row, "group": identities[row["source"]]} for row in rows]
ADAPTERS = {"fakeavceleb": import_fakeavceleb, "lavdf": import_lavdf, "avdeepfake1m": import_avdeepfake, "dfdc": import_dfdc}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
