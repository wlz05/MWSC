# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import csv
import json
import ntpath
import os
from pathlib import Path
import tempfile

def relative_path(value, must_exist=False):
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("Path must be text or a path object")
    text = os.fspath(value)
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise ValueError("Path must contain nonempty valid text")
    if ntpath.isabs(text) or Path(text).is_absolute() or ntpath.splitdrive(text)[0]:
        raise ValueError("Use a nonempty relative path: " + text)
    path = Path(text)
    if must_exist and not path.exists():
        raise FileNotFoundError(text)
    return path

def paths_match(first, second):
    left, right = Path(first).resolve(), Path(second).resolve()
    return str(left).casefold() == str(right).casefold() or left.exists() and right.exists() and left.samefile(right)

def atomic_json(path, payload):
    target = relative_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".state-", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def read_json(path):
    with relative_path(path, True).open(encoding="utf-8-sig") as stream:
        return json.load(stream)

def read_csv(path):
    with relative_path(path, True).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV requires distinct column names")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError("CSV row width differs from its header")
        return rows

def write_csv(path, rows, columns):
    target = relative_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".manifest-", dir=target.parent)
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def numpy_save(path, **arrays):
    import numpy as np
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("Feature arrays cannot contain Python objects")
    target = relative_path(path)
    if target.suffix.lower() != ".npz":
        raise ValueError("Feature archive requires an .npz suffix")
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".features-", dir=target.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
