# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import os
import pickle
import tempfile
import torch
from config import Config
from support.storage import relative_path, paths_match
from model.encoder import EncoderPair, device_from_name

FORMAT_VERSION = 1

def save_checkpoint(path, model, config, epoch, training_steps, optimizer=None, provenance="trained_locally", training=None):
    target = relative_path(path)
    config.validate()
    if type(training_steps) is not int or training_steps < 1:
        raise ValueError("Untrained encoders cannot be saved for inference")
    if type(epoch) is not int or epoch < 0 or not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("Checkpoint requires valid epoch and provenance")
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    payload = {"version": FORMAT_VERSION, "config": config.to_dict(), "model": state, "epoch": int(epoch), "training_steps": int(training_steps), "provenance": provenance}
    if training is not None:
        payload["training"] = dict(training)
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    handle, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=target.parent)
    os.close(handle)
    try:
        with open(temporary, "wb") as stream:
            torch.save(payload, stream)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def read_checkpoint(path):
    try:
        payload = torch.load(relative_path(path, True), map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as error:
        raise ValueError("Checkpoint cannot be decoded as safe tensor data") from error
    if not isinstance(payload, dict) or type(payload.get("version")) is not int or payload.get("version") != FORMAT_VERSION:
        raise ValueError("Unsupported encoder checkpoint format")
    if type(payload.get("training_steps")) is not int or payload["training_steps"] < 1:
        raise ValueError("Inference requires trained encoder weights")
    config = Config.from_dict(payload.get("config"))
    if type(payload.get("epoch")) is not int or payload["epoch"] < 0:
        raise ValueError("Checkpoint requires a nonnegative integer epoch")
    if not isinstance(payload.get("provenance"), str) or not payload["provenance"].strip():
        raise ValueError("Checkpoint lacks model provenance")
    if not isinstance(payload.get("model"), dict) or not payload["model"]:
        raise ValueError("Checkpoint requires a nonempty tensor state dictionary")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in payload["model"].items()):
        raise ValueError("Checkpoint state entries must be tensors")
    for value in payload["model"].values():
        if not torch.isfinite(value).all():
            raise ValueError("Checkpoint contains nonfinite weights")
    return payload, config

def load_encoders(path, device="auto"):
    payload, config = read_checkpoint(path)
    model = EncoderPair(config)
    model.load_state_dict(payload["model"], strict=True)
    target_device = device_from_name(device)
    model.to(target_device).eval()
    return model, config, payload

def export_encoders(checkpoint, video_path, audio_path):
    paths = [relative_path(path).resolve() for path in (checkpoint, video_path, audio_path)]
    if any(paths_match(paths[left], paths[right]) for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("Export outputs must differ from each other and the checkpoint")
    model, config, payload = load_encoders(checkpoint, "cpu")
    from core.audio import stft_features, logmel_from_stft
    import numpy as np
    video = torch.zeros(1, 1, round(config.segment_seconds * config.video_fps), config.image_size, config.image_size)
    spectrum, _ = stft_features(np.zeros(round(config.segment_seconds * config.sample_rate)), config)
    if config.architecture == "i3d_vggish":
        spectrum = logmel_from_stft(spectrum, config)
    audio = torch.from_numpy(spectrum)[None, None]
    for module, example, destination in ((model.video, video, video_path), (model.audio, audio, audio_path)):
        target = relative_path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            scripted = torch.jit.trace(module, example)
            expected = module(example)
            if not torch.allclose(scripted(example), expected, atol=1e-5, rtol=1e-4):
                raise RuntimeError("TorchScript export failed numerical validation")
            scripted.save(str(target))
    return {"video": str(video_path), "audio": str(audio_path), "config": config.to_dict(), "provenance": payload["provenance"]}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
