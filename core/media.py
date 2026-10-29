# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from dataclasses import dataclass
from numbers import Integral
import importlib.util
import json
import shutil
import subprocess
import numpy as np
from scipy.io import wavfile
from core.audio import mono_resample
from support.storage import relative_path

@dataclass
class MediaClip:
    frames: np.ndarray
    video_times: np.ndarray
    waveform: np.ndarray
    audio_start: float
    sample_rate: int
    warnings: list

    def validate(self):
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, Integral) or self.sample_rate <= 0:
            raise ValueError("Media requires a positive integer sample rate")
        frames, times, waveform = np.asarray(self.frames), np.asarray(self.video_times), np.asarray(self.waveform)
        if frames.ndim != 3 or not len(frames) or min(frames.shape[1:]) < 1:
            raise ValueError("Media frames must have shape [time, height, width]")
        if times.ndim != 1 or len(times) != len(frames) or not np.issubdtype(times.dtype, np.number) or np.iscomplexobj(times) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
            raise ValueError("Media timestamps and frame counts disagree")
        if not np.issubdtype(frames.dtype, np.number) or np.iscomplexobj(frames) or not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 255:
            raise ValueError("Media frames require finite pixel values from 0 to 255")
        if waveform.ndim != 1 or not len(waveform) or not np.issubdtype(waveform.dtype, np.number) or np.iscomplexobj(waveform) or not np.isfinite(waveform).all():
            raise ValueError("Media requires finite real mono audio")
        if isinstance(self.audio_start, bool) or not isinstance(self.audio_start, (int, float, np.integer, np.floating)) or not np.isfinite(self.audio_start):
            raise ValueError("Media requires finite mono audio and its origin")
        return self

    @property
    def start(self):
        return max(float(self.video_times[0]), self.audio_start) if len(self.video_times) else self.audio_start

    @property
    def end(self):
        if not len(self.video_times):
            return self.audio_start
        frame_step = float(np.median(np.diff(self.video_times))) if len(self.video_times) > 1 else 0.0
        return min(float(self.video_times[-1] + frame_step), self.audio_start + len(self.waveform) / self.sample_rate)

def resize_gray(frame, size):
    import cv2
    frame = np.asarray(frame)
    if frame.ndim not in (2, 3) or min(frame.shape[:2]) < 1 or frame.ndim == 3 and frame.shape[2] not in (1, 3, 4):
        raise ValueError("Image requires gray, BGR, or BGRA pixels")
    if not np.issubdtype(frame.dtype, np.number) or np.iscomplexobj(frame) or not np.isfinite(frame).all() or frame.min() < 0 or frame.max() > 255:
        raise ValueError("Image pixels must be finite and between 0 and 255")
    frame = frame.astype(np.uint8)
    if frame.ndim == 3:
        frame = frame[:, :, 0] if frame.shape[2] == 1 else cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)

def read_video_cv(path, config, start, duration, origin=0.0):
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Video cannot be opened: " + str(path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate")
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if duration is None and frame_count / fps > config.maximum_clip_seconds + 0.1:
            raise ValueError("Clip exceeds the decode limit; specify a start and duration")
        first = int(np.ceil(start * fps - 1e-8))
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        index = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        end = start + (duration if duration is not None else config.maximum_clip_seconds)
        frames, times = [], []
        while index / fps < end - 1e-8:
            ok, frame = capture.read()
            if not ok:
                break
            if index / fps >= start - 1e-8:
                frames.append(resize_gray(frame, config.image_size))
                times.append(origin + index / fps)
            index += 1
        return np.asarray(frames, dtype=np.uint8), np.asarray(times)
    finally:
        capture.release()

def read_container_av(path, config, start, duration):
    import av
    with av.open(str(path)) as container:
        videos = list(container.streams.video)
        audios = list(container.streams.audio)
        if not videos or not audios:
            raise ValueError("Container requires both video and audio streams")
        video = videos[0]
        origin = float(video.start_time * video.time_base) if video.start_time is not None else 0.0
        begin = origin + start
        limit = begin + (duration if duration is not None else config.maximum_clip_seconds)
        if duration is None and container.duration is not None and container.duration / 1000000 > config.maximum_clip_seconds + 0.1:
            raise ValueError("Clip exceeds the decode limit; specify a duration")
        if start > 0:
            container.seek(int(begin / video.time_base), stream=video, backward=True)
        frames, times = [], []
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError("Video frame has no presentation timestamp")
            timestamp = float(frame.pts * frame.time_base)
            if timestamp >= limit - 1e-8:
                break
            if timestamp >= begin - 1e-8:
                frames.append(resize_gray(frame.to_ndarray(format="bgr24"), config.image_size))
                times.append(timestamp)
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        if start > 0:
            container.seek(int(max(0.0, begin - 0.2) / stream.time_base), stream=stream, backward=True)
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=config.sample_rate)
        chunks, positions = [], []
        def collect(converted):
            if converted.pts is None:
                raise ValueError("Audio frame has no presentation timestamp")
            position = float(converted.pts * converted.time_base)
            values = converted.to_ndarray().reshape(-1)
            left = max(0, int(np.ceil((begin - position) * config.sample_rate - 1e-6)))
            right = min(len(values), int(np.ceil((limit - position) * config.sample_rate - 1e-6)))
            if right > left:
                chunks.append(values[left:right])
                positions.append(position + left / config.sample_rate)
        for frame in container.decode(audio=0):
            if frame.pts is None:
                raise ValueError("Audio frame has no presentation timestamp")
            timestamp = float(frame.pts * frame.time_base)
            for converted in resampler.resample(frame):
                collect(converted)
            if timestamp >= limit + 0.1:
                break
        for converted in resampler.resample(None):
            collect(converted)
    if not chunks:
        raise ValueError("No audio in the requested interval")
    for index in range(1, len(chunks)):
        expected = positions[index - 1] + len(chunks[index - 1]) / config.sample_rate
        if abs(positions[index] - expected) > 2 / config.sample_rate:
            raise ValueError("Audio timestamp discontinuity requires explicit packet handling")
    return MediaClip(np.asarray(frames, dtype=np.uint8), np.asarray(times), np.concatenate(chunks).astype(np.float32), positions[0], config.sample_rate, [])

def probe_stream_origins(path):
    executable = shutil.which("ffprobe")
    if executable is None:
        raise RuntimeError("Install PyAV or place ffmpeg and ffprobe on PATH")
    try:
        result = subprocess.run([executable, "-v", "error", "-show_streams", "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=60)
    except subprocess.SubprocessError as error:
        raise ValueError("FFprobe could not inspect the media") from error
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict) or not isinstance(payload.get("streams"), list):
        raise ValueError("FFprobe returned invalid stream metadata")
    streams = payload["streams"]
    origins = {}
    for stream in streams:
        if not isinstance(stream, dict):
            raise ValueError("FFprobe stream entries must be objects")
        kind = stream.get("codec_type")
        if kind in ("audio", "video") and kind not in origins:
            origins[kind] = float(stream.get("start_time", 0) or 0)
            if not np.isfinite(origins[kind]):
                raise ValueError("Stream origin must be finite")
    if set(origins) != {"audio", "video"}:
        raise ValueError("Container requires audio and video")
    return origins

def read_ffmpeg_audio(path, config, begin, duration):
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("Install PyAV or supply a separate WAV file")
    command = [executable, "-v", "error", "-i", str(path), "-ss", str(max(0.0, begin)), "-t", str(duration), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(config.sample_rate), "-f", "f32le", "pipe:1"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, timeout=60)
    except subprocess.SubprocessError as error:
        raise ValueError("FFmpeg could not decode the audio") from error
    if len(result.stdout) % 4:
        raise ValueError("FFmpeg returned an incomplete audio sample")
    return np.frombuffer(result.stdout, dtype="<f4").copy()

def read_media(video_path, config, audio_path=None, start=0.0, duration=None, audio_offset=0.0):
    config.validate()
    video_path = relative_path(video_path, True)
    if not np.isfinite(start) or start < 0 or not np.isfinite(audio_offset):
        raise ValueError("Invalid media start or offset")
    if duration is not None and (not np.isfinite(duration) or duration <= 0 or duration > config.maximum_clip_seconds):
        raise ValueError("Invalid decode duration")
    if audio_path is None and importlib.util.find_spec("av") is not None:
        import av
        try:
            clip = read_container_av(video_path, config, start, duration)
        except av.FFmpegError as error:
            raise ValueError("PyAV could not decode the media") from error
    elif audio_path is not None:
        frames, times = read_video_cv(video_path, config, start, duration)
        try:
            rate, waveform = wavfile.read(relative_path(audio_path, True), mmap=True)
        except ValueError:
            rate, waveform = wavfile.read(relative_path(audio_path, True))
        left = max(0, int(np.ceil((start - audio_offset) * rate)))
        requested_end = start + (duration if duration is not None else config.maximum_clip_seconds)
        right = min(len(waveform), max(0, int(np.ceil((requested_end - audio_offset) * rate))))
        waveform = mono_resample(waveform[left:right], rate, config.sample_rate)
        clip = MediaClip(frames, times, waveform, audio_offset + left / rate, config.sample_rate, ["OpenCV assumes constant frame rate; use PyAV for variable frame rate"])
    else:
        origins = probe_stream_origins(video_path)
        frames, times = read_video_cv(video_path, config, start, duration, origins["video"])
        begin = max(origins["video"] + start, origins["audio"])
        span = origins["video"] + start + (duration if duration is not None else config.maximum_clip_seconds) - begin
        if span <= 0:
            raise ValueError("Requested video and audio intervals do not overlap")
        container_origin = min(origins.values())
        waveform = read_ffmpeg_audio(video_path, config, begin - container_origin, span)
        clip = MediaClip(frames, times, waveform, begin, config.sample_rate, ["FFmpeg fallback uses constant-rate video timestamps; PyAV is preferred"])
    clip.validate()
    begin, end = clip.start, clip.end
    if end <= begin:
        raise ValueError("Decoded video and audio do not overlap")
    # Crop physical coverage without matching content.
    mask = (clip.video_times >= begin - 1e-8) & (clip.video_times < end - 1e-8)
    left = max(0, int(np.ceil((begin - clip.audio_start) * clip.sample_rate - 1e-6)))
    right = min(len(clip.waveform), int(np.ceil((end - clip.audio_start) * clip.sample_rate - 1e-6)))
    clip = MediaClip(clip.frames[mask], clip.video_times[mask], clip.waveform[left:right], clip.audio_start + left / clip.sample_rate, clip.sample_rate, clip.warnings)
    if len(clip.frames) < config.minimum_samples or len(clip.waveform) < config.sample_rate / 2:
        raise ValueError("Requested interval has insufficient audio or video")
    if not np.isfinite(clip.waveform).all() or not np.all(np.diff(clip.video_times) > 0):
        raise ValueError("Invalid decoded samples or timestamps")
    return clip

def segment_clips(clip, config):
    step = config.segment_seconds * (1 - config.overlap)
    begin = clip.start
    index = 0
    while begin + config.segment_seconds <= clip.end + 1 / config.sample_rate:
        end = begin + config.segment_seconds
        mask = (clip.video_times >= begin - 1e-8) & (clip.video_times < end - 1e-8)
        left = int(round((begin - clip.audio_start) * config.sample_rate))
        right = left + round(config.segment_seconds * config.sample_rate)
        if mask.sum() >= config.minimum_samples and left >= 0 and right <= len(clip.waveform):
            yield index, MediaClip(clip.frames[mask], clip.video_times[mask], clip.waveform[left:right], begin, clip.sample_rate, list(clip.warnings))
        index += 1
        begin = clip.start + index * step

def probe_duration(path):
    path = relative_path(path, True)
    if importlib.util.find_spec("av") is not None:
        import av
        try:
            with av.open(str(path)) as container:
                if not container.streams.video:
                    raise ValueError("Container has no video stream")
                stream = container.streams.video[0]
                if stream.duration is not None:
                    duration = float(stream.duration * stream.time_base)
                    if duration > 0 and np.isfinite(duration):
                        return duration
        except av.FFmpegError as error:
            raise ValueError("PyAV could not inspect the media") from error
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frames <= 0 or not np.isfinite(fps + frames):
            raise ValueError("Cannot determine video duration")
        return frames / fps
    finally:
        capture.release()

def iter_windows(video_path, config, audio_path=None, start=0.0, duration=None, audio_offset=0.0):
    config.validate()
    if not np.isfinite(start) or start < 0 or not np.isfinite(audio_offset) or duration is not None and (not np.isfinite(duration) or duration <= 0):
        raise ValueError("Invalid window interval")
    video_duration = probe_duration(video_path)
    end = video_duration if duration is None else min(video_duration, start + duration)
    step = config.segment_seconds * (1 - config.overlap)
    count = max(0, int(np.floor((end - start - config.segment_seconds) / step + 1e-8)) + 1)
    for index in range(count):
        begin = start + index * step
        try:
            clip = read_media(video_path, config, audio_path, begin, config.segment_seconds, audio_offset)
            if clip.end - clip.start < config.segment_seconds * 0.8:
                raise ValueError("Window has insufficient shared audio-video coverage")
            yield index, begin, begin + config.segment_seconds, clip, None
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            yield index, begin, begin + config.segment_seconds, None, str(error)

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
