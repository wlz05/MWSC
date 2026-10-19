# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from pathlib import Path
import tempfile
import numpy as np

class TemporaryWorkspace:
    def __enter__(self):
        self.handle = tempfile.TemporaryDirectory(prefix="test-", dir=".")
        path = Path(self.handle.name)
        self.path = path.relative_to(Path.cwd()) if path.is_absolute() else path
        return self.path

    def __exit__(self, *args):
        self.handle.cleanup()

def synthetic_media(root, name, config, phase=0.0):
    import cv2
    from scipy.io import wavfile
    video_path = root / (name + ".avi")
    audio_path = root / (name + ".wav")
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), config.video_fps, (config.image_size, config.image_size), False)
    if not writer.isOpened():
        raise RuntimeError("MJPG test codec is unavailable")
    try:
        times = np.arange(round(config.video_fps * config.segment_seconds)) / config.video_fps
        y, x = np.mgrid[:config.image_size, :config.image_size]
        for time in times:
            center = config.image_size / 2 + config.image_size / 5 * np.sin(2 * np.pi * 2 * time + phase)
            shape = np.exp(-((x - center) ** 2 + (y - config.image_size / 2) ** 2) / 20)
            frame = np.clip(25 + 200 * shape, 0, 255).astype(np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    times = np.arange(round(config.sample_rate * config.segment_seconds)) / config.sample_rate
    envelope = 0.3 + 0.2 * np.sin(2 * np.pi * 2 * times)
    waveform = envelope * np.sin(2 * np.pi * 220 * times)
    wavfile.write(audio_path, config.sample_rate, (waveform * 32767).astype(np.int16))
    return str(video_path), str(audio_path)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
