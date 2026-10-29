# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from dataclasses import replace
from fractions import Fraction
import json
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from config import Config
from core.face import FaceReader, crop_face, validate_box, color_frames
from engine.detect import Detector
from engine.lipsync import distance_profile, profile_summary, mfcc_features, valid_runs, LipSyncScorer, add_lipsync
from main import parser_for_commands, validate_outputs, expert_arguments, run
from model.expert import SyncNet, load_expert
from support.download import download_asset, fetch_assets
from verify.fixture import TemporaryWorkspace, synthetic_media

class ExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_expert_architecture_matches_pretrained_embedding_shapes(self):
        model = SyncNet().eval()
        with torch.inference_mode():
            video = model.forward_lip(torch.zeros(1, 3, 5, 224, 224))
            audio = model.forward_aud(torch.zeros(1, 1, 13, 20))
        self.assertEqual(tuple(video.shape), (1, 1024))
        self.assertEqual(tuple(audio.shape), (1, 1024))

    def test_syncnet_distance_matches_official_padded_euclidean_equation(self):
        rng = np.random.default_rng(7)
        video, audio = rng.normal(size=(40, 16)).astype(np.float32), rng.normal(size=(40, 16)).astype(np.float32)
        padded = np.pad(audio, ((3, 3), (0, 0)))
        expected = np.asarray([np.linalg.norm(video - padded[index:index + len(video)] + 1e-6, axis=1).mean() for index in range(7)])
        np.testing.assert_allclose(distance_profile(video, audio, 3), expected, rtol=1e-6)

    def test_positive_and_negative_audio_lag_have_explicit_sign(self):
        video = np.random.default_rng(32).normal(size=(120, 64)).astype(np.float32)
        for lag in (-6, 0, 5):
            audio = np.zeros_like(video)
            if lag > 0:
                audio[lag:] = video[:-lag]
            elif lag < 0:
                audio[:lag] = video[-lag:]
            else:
                audio[:] = video
            result = profile_summary(distance_profile(video, audio, 15), 15)
            self.assertEqual(result["syncnet_offset_frames"], -lag)
            self.assertAlmostEqual(result["audio_lag_seconds"], lag / 25)

    def test_lse_confidence_and_distance_are_not_probabilities(self):
        result = profile_summary([12, 11, 2, 10, 13], 2)
        self.assertEqual(result["lse_d"], 2)
        self.assertEqual(result["lse_c"], 9)
        self.assertEqual(result["zero_offset_distance"], 2)
        self.assertEqual(result["sync_decision"], "synchronized")

    def test_high_confidence_delayed_track_is_out_of_sync(self):
        result = profile_summary([1, 12, 13, 14, 15, 16, 17], 3, tolerance=0.08)
        self.assertEqual(result["sync_decision"], "out_of_sync")
        self.assertTrue(result["search_boundary_hit"])

    def test_flat_distance_curve_is_uncertain(self):
        result = profile_summary(np.ones(7), 3)
        self.assertEqual(result["lse_c"], 0)
        self.assertEqual(result["sync_decision"], "uncertain")

    def test_invalid_embeddings_or_short_search_are_rejected(self):
        for values in (np.ones((20, 2)), np.full((40, 2), np.nan), np.ones((40, 2)) + 1j, np.full((40, 2), "bad", dtype=object)):
            with self.assertRaises(ValueError):
                distance_profile(values, values)
        with self.assertRaises(ValueError):
            distance_profile(np.ones((40, 2)), np.ones((40, 2)), True)

    def test_mfcc_has_reference_shape_energy_and_finite_values(self):
        wave = np.sin(2 * np.pi * 220 * np.arange(32000) / 16000) * 0.3
        values = mfcc_features(wave)
        self.assertEqual(values.shape, (13, 199))
        self.assertTrue(np.isfinite(values).all())
        pcm = np.rint(wave * 32768)
        emphasized = np.r_[pcm[0], pcm[1:400] - 0.97 * pcm[:399]]
        expected = np.log((np.abs(np.fft.rfft(emphasized, 512)) ** 2 / 512).sum())
        self.assertAlmostEqual(float(values[0, 0]), float(expected), places=5)

    def test_mfcc_rejects_stereo_complex_and_nonfinite_audio(self):
        for values in (np.ones((16000, 2)), np.ones(10) + 1j, np.full(10, np.inf), np.array([])):
            with self.assertRaises(ValueError):
                mfcc_features(values)

    def test_face_box_validation_and_crop_preserve_color(self):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        frame[:, :, 0] = 220
        crop = crop_face(frame, (20, 20, 70, 70))
        self.assertEqual(crop.shape, (224, 224, 3))
        self.assertGreater(float(crop[:, :, 0].mean()), float(crop[:, :, 2].mean()))
        for box in ((-1, 0, 20, 20), (10, 10, 5, 5), (0, 0, np.nan, 20)):
            with self.assertRaises(ValueError):
                validate_box(box)
        with self.assertRaises(ValueError):
            crop_face(frame, (0, 0, 500, 500))

    def test_ambiguous_face_modes_are_rejected(self):
        with self.assertRaises(ValueError):
            FaceReader((0, 0, 20, 20), True)

    def test_missing_or_multiple_faces_do_not_fabricate_valid_tracks(self):
        reader = FaceReader(cropped=True)
        reader.cropped = False
        reader.detector = SimpleNamespace(detectMultiScale=lambda *args, **kwargs: [])
        frames = [(np.zeros((64, 64, 3), dtype=np.uint8), index / 25) for index in range(40)]
        for detections in ([], [(1, 1, 30, 30), (31, 1, 30, 30)]):
            reader.detector.detectMultiScale = lambda *args, **kwargs: np.asarray(detections)
            with patch("core.face.color_frames", return_value=iter(frames)):
                _, _, mask = reader.read("data/video.avi", 0, 2)
            self.assertFalse(mask.any())

    def test_valid_track_runs_never_bridge_missing_faces(self):
        self.assertEqual(list(valid_runs([True, True, False, True, False])), [(0, 2), (3, 4)])

    def test_invalid_expert_options_fail_before_loading_weights(self):
        for options in ({"batch_size": 0}, {"vshift": True}, {"minimum_confidence": np.nan}, {"tolerance": -1}):
            with patch("engine.lipsync.load_expert") as mocked, self.assertRaises(ValueError):
                LipSyncScorer(**options)
            mocked.assert_not_called()

    def test_missing_expert_does_not_destroy_mwsc_result(self):
        result = {"decision": "fake", "scores": {"mwsc_mean": 0.3}}
        with patch("engine.lipsync.LipSyncScorer", side_effect=FileNotFoundError("Missing weights")):
            report = add_lipsync(result, "data/video.mp4")
        self.assertEqual(report["decision"], "fake")
        self.assertEqual(report["lip_sync"]["status"], "unavailable")
        self.assertIsNone(report["lip_sync"]["lse_c"])
        json.dumps(report, allow_nan=False)

    def test_expert_checkpoint_requires_all_learned_parameters(self):
        with TemporaryWorkspace() as root:
            path = root / "bad.pt"
            path.touch()
            for payload in ({}, {"bad": "value"}, {"bad": torch.tensor(float("nan"))}, {"bad": torch.ones(1)}):
                with patch("model.expert.torch.load", return_value=payload), self.assertRaises(ValueError):
                    load_expert(path, "cpu")

    def test_syncnet_outputs_cannot_overwrite_weights_or_media(self):
        parser = parser_for_commands()
        for output in ("data/video.mp4", "weights/syncnet.model"):
            args = parser.parse_args(["lipsync", "--video", "data/video.mp4", "--syncnet-weights", "weights/syncnet.model", "--output", output])
            with self.assertRaises(ValueError):
                validate_outputs(args)
        with self.assertRaises(ValueError):
            validate_outputs(parser.parse_args(["fetch", "--syncnet-weights", "data/shared", "--example-output", "data/shared"]))

    def test_standalone_lipsync_needs_no_mwsc_checkpoint(self):
        args = parser_for_commands().parse_args(["lipsync", "--video", "data/video.mp4", "--face-cropped"])
        self.assertFalse(hasattr(args, "checkpoint"))
        self.assertEqual(expert_arguments(args)["vshift"], 15)
        with patch("engine.lipsync.LipSyncScorer") as mocked:
            mocked.return_value.predict.return_value = {"lse_c": 5.5}
            self.assertEqual(run(args)["lse_c"], 5.5)

    def test_failed_download_preserves_source_tree_and_cleans_partial_file(self):
        with TemporaryWorkspace() as root:
            target = root / "model.pt"
            with patch("support.download.urllib.request.urlopen", side_effect=TimeoutError("timeout")), self.assertRaises(ValueError):
                download_asset("syncnet_v2.model", target)
            self.assertFalse(list(root.iterdir()))

    def test_download_never_replaces_existing_invalid_model(self):
        with TemporaryWorkspace() as root:
            target = root / "model.pt"
            target.write_bytes(b"original")
            with patch("support.download.urllib.request.urlopen") as mocked, self.assertRaises(ValueError):
                download_asset("syncnet_v2.model", target, lambda path: (_ for _ in ()).throw(ValueError("bad model")))
            mocked.assert_not_called()
            self.assertEqual(target.read_bytes(), b"original")

    def test_fetch_collisions_are_rejected_before_any_download(self):
        with patch("support.download.download_asset") as mocked, self.assertRaises(ValueError):
            fetch_assets("data/shared", "data/shared")
        mocked.assert_not_called()

    def test_mwsc_video_report_includes_aggregate_scores(self):
        detector = object.__new__(Detector)
        detector.config, detector.model_id = Config(), "test"
        window = {"decision": "real", "synchronization_score": 0.9, "score_upper_bound": 1.2}
        with patch("engine.detect.iter_windows", return_value=[(0, 0, 2, object(), None)]), patch.object(detector, "process_segment", return_value=window):
            report = detector.predict("data/video.mp4")
        self.assertAlmostEqual(report["scores"]["mwsc_mean"], 0.9)
        self.assertAlmostEqual(report["scores"]["mwsc_normalized_mean"], 0.75)

    def test_empty_mwsc_report_has_null_scores(self):
        detector = object.__new__(Detector)
        detector.config = Config()
        with patch("engine.detect.iter_windows", return_value=[]):
            report = detector.predict("data/short.mp4")
        self.assertIsNone(report["scores"]["mwsc_mean"])

    def test_precropped_video_runs_the_complete_lipsync_reporting_path(self):
        scorer = object.__new__(LipSyncScorer)
        scorer.vshift, scorer.model_id = 3, "stub_expert_for_pipeline_only"
        scorer.minimum_confidence, scorer.tolerance = 3.0, 0.08
        scorer.faces = FaceReader(cropped=True)
        rng = np.random.default_rng(5)
        def embeddings(frames, waveform):
            values = rng.normal(size=(len(frames) - 5, 32)).astype(np.float32)
            return values, values.copy()
        scorer.embeddings = embeddings
        with TemporaryWorkspace() as root:
            video, audio = synthetic_media(root, "color", replace(Config(), image_size=64))
            report = scorer.predict(video, audio, window_seconds=2, output=root / "score.json")
            self.assertEqual(report["status"], "scored")
            self.assertEqual(report["scored_tracks"], 1)
            self.assertAlmostEqual(report["audio_lag_seconds"], 0)
            self.assertGreater(report["coverage"], 0.9)
            json.dumps(report, allow_nan=False)

    def test_download_validation_receives_a_relative_temporary_path(self):
        with TemporaryWorkspace() as root:
            response = io.BytesIO(b"data" * 300)
            response.headers = {"Content-Length": "1200"}
            validated = []
            def validate(path):
                self.assertFalse(Path(path).is_absolute())
                self.assertEqual(Path(path).read_bytes(), b"data" * 300)
                validated.append(path)
            with patch("support.download.urllib.request.urlopen", return_value=response):
                result = download_asset("syncnet_v2.model", root / "model.pt", validate)
            self.assertEqual(result["bytes"], 1200)
            self.assertEqual(len(validated), 1)
            self.assertEqual(len(list(root.iterdir())), 1)

    def test_incomplete_http_download_is_not_committed(self):
        with TemporaryWorkspace() as root:
            response = io.BytesIO(b"data" * 300)
            response.headers = {"Content-Length": "2400"}
            with patch("support.download.urllib.request.urlopen", return_value=response), self.assertRaises(ValueError):
                download_asset("syncnet_v2.model", root / "model.pt")
            self.assertFalse(list(root.iterdir()))

    def test_profile_summary_rejects_nonfinite_or_invalid_options(self):
        for options in ({"vshift": True}, {"vshift": 2, "minimum_confidence": np.nan}, {"vshift": 2, "tolerance": -1}):
            with self.assertRaises(ValueError):
                profile_summary(np.ones(5), **options)

    def test_missing_opencv_detector_produces_an_actionable_error(self):
        import cv2
        with patch.object(cv2, "CascadeClassifier", None, create=True), patch.object(cv2, "xobjdetect", None, create=True), patch.object(cv2, "xobjdetect_CascadeClassifier", None, create=True):
            with self.assertRaisesRegex(RuntimeError, "OpenCV"):
                FaceReader()

    def test_predict_attachment_is_saved_with_mwsc_and_syncnet_scores(self):
        with TemporaryWorkspace() as root:
            args = parser_for_commands().parse_args(["predict", "--video", "data/video.mp4", "--checkpoint", "weights/mwsc.pt", "--lipsync", "--output", str(root / "result.json")])
            detector = SimpleNamespace(predict=lambda *args: {"decision": "real", "scores": {"mwsc_mean": 0.9}})
            with patch("main.make_detector", return_value=detector), patch("engine.lipsync.LipSyncScorer") as mocked:
                mocked.return_value.predict.return_value = {"status": "scored", "lse_c": 7.5, "lse_d": 5.2}
                result = run(args)
            saved = json.loads((root / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result, saved)
            self.assertEqual(saved["scores"]["mwsc_mean"], 0.9)
            self.assertEqual(saved["lip_sync"]["lse_c"], 7.5)

    def test_legacy_batch_counters_are_the_only_optional_checkpoint_fields(self):
        with TemporaryWorkspace() as root:
            path = root / "legacy.pt"
            path.touch()
            model = SyncNet()
            state = {key: value for key, value in model.state_dict().items() if not key.endswith("num_batches_tracked")}
            with patch("model.expert.torch.load", return_value=state):
                loaded = load_expert(path, "cpu")
            self.assertFalse(loaded.training)
            self.assertTrue(torch.equal(loaded.netfcaud[0].weight, model.netfcaud[0].weight))

    def test_direct_lipsync_output_cannot_overwrite_media_or_weights(self):
        scorer = object.__new__(LipSyncScorer)
        scorer.weights = "weights/syncnet.model"
        for output in ("data/video.mp4", "data/audio.wav", "weights/syncnet.model"):
            with self.assertRaises(ValueError):
                scorer.predict("data/video.mp4", "data/audio.wav", output=output)

    def test_mfcc_matches_independent_reference_cepstral_coefficients(self):
        wave = np.sin(2 * np.pi * 220 * np.arange(32000) / 16000) * 0.3
        expected = [18.17652620799456, 21.58376739447453, 12.268425020840379, 6.251668778687186, 0.24334545173186878, -6.76610124675194, -12.867269215646898, -18.891673599445607, -22.862848550772256, -25.14365552031895, -24.067686811338902, -21.80990057995982, -17.949747656094576]
        np.testing.assert_allclose(mfcc_features(wave)[:, 0], expected, rtol=1e-6, atol=1e-6)

    def test_fresh_lipsync_install_selects_a_compatible_opencv_build(self):
        import install
        with patch("install.sys.argv", ["install.py", "--install", "--lipsync"]), patch("install.importlib.util.find_spec", side_effect=lambda name: None if name == "cv2" else object()), patch("install.subprocess.run") as mocked, patch("builtins.print"):
            install.main()
        self.assertIn("opencv-python>=4.8,<5", mocked.call_args.args[0])

    def test_color_pts_preserve_container_origin_or_normalize_external_wav(self):
        stream = SimpleNamespace(start_time=225, time_base=Fraction(1, 25))
        frames = [SimpleNamespace(pts=225 + index, time_base=Fraction(1, 25), to_ndarray=lambda **kwargs: np.zeros((32, 32, 3), dtype=np.uint8)) for index in range(2)]
        fake_av = SimpleNamespace(FFmpegError=RuntimeError)
        with TemporaryWorkspace() as root:
            path = root / "video.avi"
            path.touch()
            for origin, expected in ((None, 9.0), (0.0, 0.0)):
                container = SimpleNamespace(streams=SimpleNamespace(video=[stream]), decode=lambda **kwargs: iter(frames))
                from unittest.mock import MagicMock
                context = MagicMock()
                context.__enter__.return_value = container
                fake_av.open = lambda *args: context
                with patch("core.face.importlib.util.find_spec", return_value=object()), patch.dict("sys.modules", {"av": fake_av}):
                    result = list(color_frames(path, 0, 1, origin))
                self.assertAlmostEqual(result[0][1], expected)
                self.assertAlmostEqual(result[1][1] - result[0][1], 0.04)

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
