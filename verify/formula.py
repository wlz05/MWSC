# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from dataclasses import replace
import unittest
import numpy as np
from config import Config
from core.wavelet import first_component, morlet_cwt, scale_coherence, gaussian_weights, synchronization_score, InsufficientEvidence
from core.threshold import DynamicThreshold
from verify.fixture import TemporaryWorkspace

class WaveletTests(unittest.TestCase):
    def test_coherence_matches_complex_equation(self):
        video = np.array([[1 + 2j, 3 - 1j, 2j], [2, 1j, 3]])
        audio = np.array([[2j, 1 + 3j, 4], [1j, 2, -1]])
        expected = []
        for left, right in zip(video, audio):
            expected.append(abs(sum(left * right.conjugate())) ** 2 / (sum(abs(left) ** 2) * sum(abs(right) ** 2)))
        np.testing.assert_allclose(scale_coherence(video, audio), expected)

    def test_cancellation_occurs_before_square(self):
        value = scale_coherence(np.array([[1, 1]]), np.array([[1, -1]]))
        self.assertEqual(float(value[0]), 0)

    def test_constant_phase_is_invariant(self):
        rng = np.random.default_rng(3)
        values = rng.normal(size=(8, 60)) + 1j * rng.normal(size=(8, 60))
        phase_shifted = values * np.exp(1j * 1.2)
        np.testing.assert_allclose(scale_coherence(values, phase_shifted), 1, atol=1e-12)

    def test_zero_wavelet_energy_is_rejected(self):
        with self.assertRaises(InsufficientEvidence):
            scale_coherence(np.zeros((2, 10)), np.ones((2, 10)))

    def test_pca_recovers_rank_one_temporal_series(self):
        signal = np.linspace(-1, 1, 60)
        features = signal[:, None] * np.array([[1, 4, -2]]) + 12
        recovered, explained = first_component(features)
        self.assertAlmostEqual(explained, 1)
        self.assertGreater(abs(np.corrcoef(signal, recovered)[0, 1]), 0.999999)

    def test_constant_features_are_rejected(self):
        with self.assertRaises(InsufficientEvidence):
            first_component(np.ones((60, 4)))

    def test_morlet_resolves_frequency(self):
        times = np.arange(900) / 30
        signal = np.sin(2 * np.pi * 3 * times)
        frequencies = np.array([1.0, 2.0, 3.0, 4.0])
        wave, scales, boundary = morlet_cwt(signal, 30, frequencies)
        energy = np.mean(abs(wave[:, 200:-200]) ** 2, axis=1)
        self.assertEqual(int(np.argmax(energy)), 2)
        self.assertTrue(np.all(np.diff(scales) < 0))
        self.assertTrue(np.all((boundary >= 0) & (boundary <= 1)))

    def test_raw_and_normalized_weights_are_explicit(self):
        frequencies = np.linspace(0.5, 4, 8)
        raw = gaussian_weights(frequencies)
        expected = np.exp(-(frequencies - 4) ** 2 / (2 * 1.5 ** 2)) / (1.5 * np.sqrt(2 * np.pi))
        np.testing.assert_allclose(raw, expected)
        self.assertGreater(raw.sum(), 1)
        self.assertAlmostEqual(gaussian_weights(frequencies, mode="normalized").sum(), 1)

    def test_identical_features_reach_weight_sum(self):
        rng = np.random.default_rng(42)
        features = rng.normal(size=(60, 5))
        result = synchronization_score(features, features, Config())
        self.assertAlmostEqual(result["synchronization_score"], result["score_upper_bound"], places=10)

    def test_invalid_configuration_is_rejected(self):
        for values in ({"overlap": 1}, {"gaussian_std": 0}, {"video_fps": 8}, {"weight_mode": "implicit"}):
            with self.assertRaises(ValueError):
                Config.from_dict(values)

class ThresholdTests(unittest.TestCase):
    def test_fake_never_updates_history(self):
        tracker = DynamicThreshold(Config())
        result = tracker.decide(0.4)
        self.assertEqual(result["decision"], "fake")
        self.assertEqual(tracker.threshold, 0.75)
        self.assertEqual(len(tracker.history), 0)

    def test_real_updates_next_threshold(self):
        tracker = DynamicThreshold(Config())
        result = tracker.decide(0.9)
        self.assertEqual(result["threshold_before"], 0.75)
        self.assertAlmostEqual(result["threshold_after"], 0.9 * 0.75 + 0.1 * 0.9)

    def test_exact_ema_and_sliding_history(self):
        config = replace(Config(), threshold=0.1, history_size=2)
        tracker = DynamicThreshold(config)
        expected = config.threshold
        history = []
        for score in (0.8, 0.9, 0.85):
            history = (history + [score])[-2:]
            expected = 0.9 * expected + 0.1 * (np.mean(history) - 2 * np.std(history))
            tracker.decide(score)
            self.assertAlmostEqual(tracker.threshold, expected)
        self.assertEqual(list(tracker.history), [0.9, 0.85])

    def test_equality_is_real_and_frozen_has_no_mutation(self):
        tracker = DynamicThreshold(Config())
        before = tracker.state_dict()
        self.assertEqual(tracker.decide(0.75, False)["decision"], "real")
        self.assertEqual(tracker.state_dict(), before)

    def test_state_is_bound_to_config_stream_and_model(self):
        with TemporaryWorkspace() as root:
            path = root / "state.json"
            tracker = DynamicThreshold(Config(), "meeting", "encoder-a")
            tracker.decide(0.9)
            tracker.save(path)
            restored = DynamicThreshold(Config(), "meeting", "encoder-a").load(path)
            self.assertEqual(restored.state_dict(), tracker.state_dict())
            for candidate in (DynamicThreshold(Config(), "other", "encoder-a"), DynamicThreshold(Config(), "meeting", "encoder-b"), DynamicThreshold(replace(Config(), weight_mode="normalized"), "meeting", "encoder-a")):
                with self.assertRaises(ValueError):
                    candidate.load(path)

    def test_nonfinite_score_is_rejected(self):
        with self.assertRaises(ValueError):
            DynamicThreshold(Config()).decide(float("nan"))

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
