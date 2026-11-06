# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import csv
import json
import math
import ntpath
import os
from pathlib import Path
import tempfile
import shutil

def relative_path(value, must_exist=False):
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("Path must be text or a path object")
    text = os.fspath(value)
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise ValueError("Path must contain nonempty valid text")
    if text.startswith(("/", "\\")) or ntpath.isabs(text) or Path(text).is_absolute() or ntpath.splitdrive(text)[0]:
        raise ValueError("Use a nonempty relative path: " + text)
    path = Path(text)
    if must_exist and not path.exists():
        raise FileNotFoundError(text)
    return path

def storage_status(path, minimum_free_gb=5.0):
    target = relative_path(path)
    if isinstance(minimum_free_gb, bool) or not isinstance(minimum_free_gb, (int, float)) or not math.isfinite(minimum_free_gb) or minimum_free_gb < 0:
        raise ValueError("Free-space reserve must be finite and nonnegative")
    if target.exists() and not target.is_dir():
        raise ValueError("Storage root must be a directory")
    inspected = target
    while not inspected.exists():
        inspected = inspected.parent
    if not inspected.is_dir():
        raise ValueError("Storage parent must be a directory")
    usage = shutil.disk_usage(inspected)
    return {"path": target.as_posix(), "exists": target.exists(),
        "inspected_path": inspected.as_posix(),
        "filesystem_id": int(inspected.stat().st_dev), "total_gb": usage.total / 1e9,
        "free_gb": usage.free / 1e9, "minimum_free_gb": float(minimum_free_gb),
        "reserve_available": usage.free / 1e9 >= minimum_free_gb}

def paths_match(first, second):
    left, right = Path(first).resolve(), Path(second).resolve()
    return str(left).casefold() == str(right).casefold() or left.exists() and right.exists() and left.samefile(right)

def protect_output(path, inputs):
    if path is None:
        return None
    target = relative_path(path)
    for value in inputs:
        if value is not None and value != "" and paths_match(target, relative_path(value)):
            raise ValueError("Output path would overwrite an input file")
    return target

def atomic_json(path, payload):
    target = relative_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".state-", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        # Replace the target only after serialization succeeds.
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def read_json(path):
    def finite_number(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON numbers must be finite")
        return number
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field: " + key)
            result[key] = value
        return result
    with relative_path(path, True).open(encoding="utf-8-sig") as stream:
        return json.load(stream, object_pairs_hook=unique_object, parse_float=finite_number, parse_constant=finite_number)

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

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
