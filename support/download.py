# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import hashlib
import http.client
import os
import tempfile
import urllib.error
import urllib.request
from support.storage import relative_path, paths_match

BASE_URL = "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/"

def download_asset(name, destination, validator=None):
    if name not in ("syncnet_v2.model", "example.avi"):
        raise ValueError("Unknown official SyncNet asset")
    target = relative_path(destination)
    if target.exists():
        if validator is not None:
            validator(target)
        return {"path": str(target), "cached": True, "bytes": target.stat().st_size, "sha256": digest_file(target)}
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".download-", dir=target.parent)
    size = 0
    try:
        with os.fdopen(handle, "wb") as stream:
            try:
                with urllib.request.urlopen(BASE_URL + name, timeout=30) as response:
                    if "text/html" in response.headers.get("Content-Type", ""):
                        raise ValueError("Asset server returned HTML instead of model data")
                    expected = response.headers.get("Content-Length")
                    while True:
                        chunk = response.read(1048576)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > 268435456:
                            raise ValueError("Asset exceeds the 256 MiB download limit")
                        stream.write(chunk)
                    if expected is not None and (not expected.isdigit() or size != int(expected)):
                        raise ValueError("Downloaded asset length differs from its header")
            except (urllib.error.URLError, TimeoutError, http.client.HTTPException) as error:
                raise ValueError("Official SyncNet download failed: " + str(error)) from error
        if size < 1024:
            raise ValueError("Downloaded asset is incomplete")
        if validator is not None:
            validator(os.path.relpath(temporary))
        if target.exists():
            raise ValueError("Download destination appeared during the transfer")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(target), "cached": False, "bytes": size, "sha256": digest_file(target)}

def digest_file(path):
    digest = hashlib.sha256()
    with relative_path(path, True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()

def fetch_assets(weights, example=None):
    from model.expert import load_expert
    if example is not None and paths_match(weights, example):
        raise ValueError("SyncNet weights and example paths must differ")
    result = {"weights": download_asset("syncnet_v2.model", weights, lambda path: load_expert(path, "cpu"))}
    if example is not None:
        from core.media import probe_duration
        result["example"] = download_asset("example.avi", example, probe_duration)
    return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
