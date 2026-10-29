# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from collections import deque
import numpy as np
from core.media import MediaClip, resize_gray

class StreamSession:
    def __init__(self, detector):
        self.detector = detector
        self.config = detector.config
        self.frames = deque()
        self.audio = deque()
        self.next_start = None
        self.last_video_time = None
        self.last_audio_end = None

    def push_video(self, frame, timestamp):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or self.last_video_time is not None and timestamp <= self.last_video_time:
            raise ValueError("Video packets must have increasing finite timestamps")
        if self.frames and timestamp - self.frames[0][0] > self.config.maximum_clip_seconds:
            raise BufferError("Consume ready segments before buffering more video")
        if len(self.frames) >= int(np.ceil(self.config.maximum_clip_seconds * self.config.video_fps)) + 1:
            raise BufferError("Video packet count exceeds the configured buffer budget")
        self.frames.append((timestamp, resize_gray(np.asarray(frame), self.config.image_size)))
        self.last_video_time = timestamp

    def push_audio(self, waveform, timestamp):
        timestamp = float(timestamp)
        values = np.asarray(waveform, dtype=np.float32)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not np.isfinite(timestamp):
            raise ValueError("Audio packet must contain finite mono samples")
        if len(self.audio) >= int(np.ceil(self.config.maximum_clip_seconds * 1000 / self.config.hop_ms)) + 1:
            raise BufferError("Audio packet count exceeds the configured buffer budget")
        if self.last_audio_end is not None and abs(timestamp - self.last_audio_end) > 2 / self.config.sample_rate:
            raise ValueError("Audio packets contain a gap or overlap")
        timestamp = self.last_audio_end if self.last_audio_end is not None else timestamp
        end = timestamp + len(values) / self.config.sample_rate
        origin = self.audio[0][0] if self.audio else timestamp
        if end - origin > self.config.maximum_clip_seconds:
            raise BufferError("Consume ready segments before buffering more audio")
        # Snap accepted packet jitter to the PCM timeline.
        self.audio.append((timestamp, values.copy()))
        self.last_audio_end = timestamp + len(values) / self.config.sample_rate

    def drain(self, update=True):
        results = []
        if not self.frames or not self.audio:
            return results
        if self.next_start is None:
            self.next_start = max(self.frames[0][0], self.audio[0][0])
        step = self.config.segment_seconds * (1 - self.config.overlap)
        while self.last_video_time + 1 / self.config.video_fps >= self.next_start + self.config.segment_seconds - 1e-8 and self.last_audio_end >= self.next_start + self.config.segment_seconds - 1e-8:
            start = self.next_start
            end = start + self.config.segment_seconds
            observations = [(time, frame) for time, frame in self.frames if start <= time < end - 1e-8]
            audio_start = self.audio[0][0]
            waveform = np.concatenate([chunk for _, chunk in self.audio])
            left = round((start - audio_start) * self.config.sample_rate)
            right = left + round(self.config.segment_seconds * self.config.sample_rate)
            if observations and left >= 0 and right <= len(waveform):
                times, frames = zip(*observations)
                clip = MediaClip(np.stack(frames), np.asarray(times), waveform[left:right], start, self.config.sample_rate, [])
                results.append(self.detector.process_segment(clip, update))
            else:
                results.append({"decision": "insufficient_evidence", "reason": "Missing stream segment", "segment_start": start, "segment_end": end, "updated": False})
            self.next_start += step
            while self.frames and self.frames[0][0] < self.next_start - 1e-8:
                self.frames.popleft()
            while self.audio and self.audio[0][0] + len(self.audio[0][1]) / self.config.sample_rate <= self.next_start + 1e-8:
                self.audio.popleft()
        return results

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
