# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from dataclasses import replace
import unittest
import numpy as np
from config import Config
from core.align import timestamp_dtw, align_features, AlignmentError
from core.audio import preemphasize, mono_resample, stft_features, logmel_from_stft
from core.media import read_media, segment_clips
from engine.stream import StreamSession
from verify.fixture import TemporaryWorkspace, synthetic_media

class AlignmentTests(unittest.TestCase):
    def test_dtw_is_monotonic_and_bounded(self):
        video = np.arange(60) / 30
        audio = np.arange(200) / 100
        corrected, pairs = timestamp_dtw(video, audio)
        self.assertTrue(np.all(np.diff(corrected) > 0))
        self.assertLessEqual(np.max(abs(corrected - video)), 0.05)
        self.assertTrue(np.all(np.diff(pairs, axis=0) >= 0))
        self.assertEqual(pairs[0].tolist(), [0, 0])
        self.assertEqual(pairs[-1].tolist(), [59, 199])

    def test_large_offset_is_not_content_warped_away(self):
        video = np.arange(60) / 30
        audio = np.arange(200) / 100 + 0.2
        with self.assertRaises(AlignmentError):
            timestamp_dtw(video, audio)

    def test_repeated_timestamps_are_rejected(self):
        with self.assertRaises(AlignmentError):
            timestamp_dtw([0, 0, 1], [0, 0.5, 1])

    def test_sparse_features_fail_nyquist(self):
        config = replace(Config(), maximum_gap=1.0)
        times = np.arange(10) / 5
        with self.assertRaises(AlignmentError):
            align_features(np.ones((10, 2)), np.ones((10, 2)), times, times, config)

    def test_features_share_a_common_grid(self):
        video = np.arange(60) / 30
        audio = np.arange(200) / 100
        left, right, grid, diagnostics = align_features(video[:, None], audio[:, None], video, audio, Config())
        self.assertEqual(left.shape, right.shape)
        self.assertEqual(len(grid), diagnostics["aligned_samples"])
        self.assertLess(np.max(abs(left - right)), 0.006)

class AudioTests(unittest.TestCase):
    def test_preemphasis_filter(self):
        np.testing.assert_allclose(preemphasize([1, 2, 3], 0.97), [1, 1.03, 1.06])

    def test_pcm_normalization_and_resampling(self):
        values = np.full(8000, 16384, dtype=np.int16)
        result = mono_resample(values, 8000, 16000)
        self.assertEqual(len(result), 16000)
        self.assertAlmostEqual(float(result[100:-100].mean()), 0.5, places=3)

    def test_unsigned_pcm_is_centered(self):
        result = mono_resample(np.full(100, 128, dtype=np.uint8), 16000, 16000)
        np.testing.assert_allclose(result, 0)

    def test_stft_and_mel_dimensions(self):
        config = Config()
        waveform = np.sin(2 * np.pi * 220 * np.arange(32000) / 16000)
        spectrum, times = stft_features(waveform, config)
        self.assertEqual(spectrum.shape, (257, 198))
        self.assertEqual(logmel_from_stft(spectrum, config).shape, (64, 198))
        self.assertAlmostEqual(float(np.diff(times).mean()), 0.01)

class MediaTests(unittest.TestCase):
    def test_separate_wav_decode_and_segment(self):
        with TemporaryWorkspace() as root:
            config = replace(Config(), image_size=32)
            video, audio = synthetic_media(root, "clip", config)
            clip = read_media(video, config, audio)
            segments = list(segment_clips(clip, config))
            self.assertEqual(clip.frames.shape, (60, 32, 32))
            self.assertEqual(len(clip.waveform), 32000)
            self.assertEqual(len(segments), 1)

    def test_streaming_emits_each_overlapping_window_once(self):
        class DetectorStub:
            config = replace(Config(), image_size=32)
            def process_segment(self, clip, update):
                return {"segment_start": clip.start, "segment_end": clip.end}
        session = StreamSession(DetectorStub())
        for index in range(90):
            session.push_video(np.zeros((32, 32), dtype=np.uint8), index / 30)
        session.push_audio(np.ones(48000, dtype=np.float32), 0)
        results = session.drain()
        self.assertEqual([item["segment_start"] for item in results], [0.0, 1.0])
        self.assertEqual(session.drain(), [])

    def test_audio_packet_gap_is_rejected(self):
        class DetectorStub:
            config = Config()
        session = StreamSession(DetectorStub())
        session.push_audio(np.ones(1600), 0)
        with self.assertRaises(ValueError):
            session.push_audio(np.ones(1600), 0.2)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
