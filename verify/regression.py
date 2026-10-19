# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from dataclasses import replace
import json
import os
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from config import Config
from core.align import align_features, interpolate_features, validate_times, AlignmentError
from model.extract import prepare_inputs
from dataset.manifest import validate_rows, select_rows, parse_segments, window_label
from core.media import MediaClip, iter_windows, read_media, read_ffmpeg_audio, probe_stream_origins
from dataset.speech import import_speech, audit_media
from engine.sync import estimate_offset, shifted_clip, evaluate_sync
from engine.stream import StreamSession
from core.threshold import DynamicThreshold
from core.audio import mono_resample
from dataset.split import connected_groups
from engine.evaluate import evaluate_row
from engine.detect import Detector
from support.storage import atomic_json, relative_path, read_csv, write_csv, numpy_save
from support.metrics import binary_metrics
from model.weights import read_checkpoint, export_encoders
from dataset.samples import RealSegmentDataset
from main import load_config, parser_for_commands, validate_outputs
from engine.train import train
from dataset.adapter import import_avdeepfake
from verify.fixture import TemporaryWorkspace, synthetic_media
from core.wavelet import InsufficientEvidence, gaussian_weights, first_component

class ConfigurationTests(unittest.TestCase):
    def test_fractional_integer_and_boolean_parameters_are_rejected(self):
        for parameters in ({"frequency_count": 2.5}, {"sample_rate": True}, {"gaussian_std": "1.5"}):
            with self.assertRaises(ValueError):
                Config.from_dict(parameters)

    def test_configuration_requires_dictionary(self):
        with self.assertRaises(ValueError):
            Config.from_dict([])

class SynchronizationTests(unittest.TestCase):
    def test_positive_and_negative_offsets_have_correct_sign(self):
        config = Config()
        rng = np.random.default_rng(18)
        original = rng.normal(size=180)
        times = np.arange(180) / config.video_fps
        video = np.column_stack((original, 2 * original))
        for lag in (-6, 0, 9):
            shifted = np.zeros_like(original)
            if lag > 0:
                shifted[lag:] = original[:-lag]
            elif lag < 0:
                shifted[:lag] = original[-lag:]
            else:
                shifted = original.copy()
            audio = np.column_stack((shifted, -3 * shifted))
            result = estimate_offset(video, audio, times, times, config)
            self.assertEqual(result["estimated_offset_frames"], lag)
            self.assertAlmostEqual(result["estimated_offset_seconds"], lag / 30)

    def test_content_shift_crops_padding_and_retains_timeline(self):
        frames = np.zeros((60, 32, 32), dtype=np.uint8)
        waveform = np.arange(32000, dtype=np.float32)
        clip = MediaClip(frames, np.arange(60) / 30, waveform, 0.0, 16000, [])
        delayed = shifted_clip(clip, 0.3)
        self.assertAlmostEqual(delayed.audio_start, 0.3)
        self.assertEqual(delayed.waveform[0], 0)
        self.assertEqual(len(delayed.waveform), 27200)
        advanced = shifted_clip(clip, -0.3)
        self.assertEqual(advanced.waveform[0], 4800)
        self.assertAlmostEqual(advanced.audio_start, 0)
        self.assertLess(advanced.video_times[-1], 1.7)

    def test_authenticity_and_synchronization_labels_are_independent(self):
        row = {"dataset": "custom", "video": "data/real.mp4", "label": 0, "group": "one", "source": "one", "split": "train", "sync_label": 1, "offset_seconds": 0.3}
        rows = validate_rows([row])
        self.assertEqual(rows[0]["label"], 0)
        self.assertEqual(rows[0]["sync_label"], 1)
        with self.assertRaises(ValueError):
            select_rows(rows, "train", True)

    def test_unknown_sync_label_is_not_inferred_from_fake_label(self):
        row = {"dataset": "custom", "video": "data/fake.mp4", "label": 1, "group": "one", "source": "one"}
        self.assertEqual(validate_rows([row])[0]["sync_label"], "")

class DatasetRegressionTests(unittest.TestCase):
    def test_shared_namespace_links_related_datasets(self):
        rows = [{"dataset": "first", "source": "a", "group": "person", "identity_namespace": "shared"}, {"dataset": "second", "source": "b", "group": "person", "identity_namespace": "shared"}]
        self.assertEqual(len(connected_groups(rows)), 1)

    def test_audio_and_visual_intervals_survive_import(self):
        with TemporaryWorkspace() as root:
            path = root / "metadata.json"
            atomic_json(path, [{"file": "test/id0001/fake.mp4", "original": "test/id0001/real.mp4", "split": "test", "modify_type": "both_modified", "audio_fake_segments": [[0.2, 0.5]], "visual_fake_segments": [[1.0, 1.5]]}])
            row = import_avdeepfake(root / "data", path)[0]
            self.assertIn("0.2", row["audio_segments"])
            self.assertIn("1.0", row["video_segments"])
            self.assertIn("1.5", row["fake_segments"])

    def test_speech_filelist_and_wav_pairing(self):
        with TemporaryWorkspace() as root:
            video = root / "video" / "id0001" / "talk" / "clip.mp4"
            audio = root / "audio" / "id0001" / "talk" / "clip.wav"
            video.parent.mkdir(parents=True)
            audio.parent.mkdir(parents=True)
            video.touch()
            audio.touch()
            filelist = root / "train.txt"
            filelist.write_text("id0001/talk/clip\n", encoding="ascii")
            rows = import_speech(root / "video", "voxceleb2", filelist, "train", root / "audio")
            self.assertEqual(rows[0]["group"], "id0001")
            self.assertEqual(rows[0]["sync_label"], "")
            self.assertEqual(rows[0]["audio"], str(audio).replace("\\", "/"))

class MediaRegressionTests(unittest.TestCase):
    def test_native_low_cadence_is_rejected_before_upsampling(self):
        config = replace(Config(), image_size=32)
        clip = MediaClip(np.zeros((10, 32, 32), dtype=np.uint8), np.arange(10) / 5, np.ones(32000, dtype=np.float32), 0, 16000, [])
        with self.assertRaises(InsufficientEvidence):
            prepare_inputs(clip, config)

    def test_zero_frame_clip_has_finite_interval(self):
        clip = MediaClip(np.empty((0, 32, 32)), np.array([]), np.ones(16000), 0, 16000, [])
        self.assertEqual(clip.start, 0)
        self.assertEqual(clip.end, 0)

    def test_unequal_feature_tail_is_trimmed_inside_tolerance(self):
        config = replace(Config(), maximum_gap=0.2)
        video_times = np.arange(20) / 10
        audio_times = np.arange(198) / 100 + 0.012
        video, audio, grid, _ = align_features(video_times[:, None], audio_times[:, None], video_times, audio_times, config)
        self.assertEqual(len(video), len(audio))
        self.assertGreater(len(grid), 40)

    def test_bad_video_does_not_abort_dataset_evaluation(self):
        class DetectorStub:
            config = Config()
        row = {"video": "data/missing.mp4", "duration": ""}
        result = evaluate_row(DetectorStub(), row)
        self.assertEqual(result["decision"], "insufficient_evidence")
        self.assertEqual(result["valid_segments"], 0)

    def test_long_video_uses_bounded_decode_windows(self):
        config = replace(Config(), image_size=32, maximum_clip_seconds=2)
        fixture_config = replace(config, segment_seconds=5, maximum_clip_seconds=5)
        with TemporaryWorkspace() as root:
            video, audio = synthetic_media(root, "long", fixture_config)
            windows = list(iter_windows(video, config, audio))
            self.assertEqual(len(windows), 4)
            self.assertTrue(all(item[4] is None for item in windows))
            self.assertTrue(all(len(item[3].waveform) == 32000 for item in windows))

    def test_media_audit_reports_failures_and_successes(self):
        config = replace(Config(), image_size=32)
        with TemporaryWorkspace() as root:
            video, audio = synthetic_media(root, "valid", config)
            report = audit_media([{"video": video, "audio": audio, "audio_offset": 0}, {"video": "data/missing.mp4", "audio": "", "audio_offset": 0}], config)
            self.assertEqual(report["passed"], 1)
            self.assertEqual(report["failed"], 1)

class BoundaryTests(unittest.TestCase):
    def row(self, **overrides):
        values = {"dataset": "private", "video": "data/example.mp4", "label": 0, "group": "alice", "source": "session", "split": "test", "duration": 2}
        values.update(overrides)
        return values

    def test_fractional_and_boolean_manifest_labels_are_rejected(self):
        for field in ("label", "sync_label"):
            for value in (0.5, 1.5, True, False, "0.5"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_rows([self.row(**{field: value})])

    def test_every_modal_interval_is_checked_against_duration(self):
        for field in ("audio_segments", "video_segments"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_rows([self.row(label=1, fake_segments="[[0.1,0.2]]", **{field: "[[0.1,20]]"})])

    def test_invalid_interval_containers_are_rejected(self):
        for value in ("null", "{}", "123", '"12"', "[null]", "[1]", "[[0,null]]"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_segments(value)

    def test_zero_duration_and_empty_identity_are_rejected(self):
        for values in ({"duration": 0}, {"group": "; ;"}, {"label": None}):
            with self.assertRaises(ValueError):
                validate_rows([self.row(**values)])

    def test_unrepresentable_sampling_steps_are_rejected(self):
        for values in ({"hop_ms": 0.001}, {"overlap": 1 - 1e-15}, {"segment_seconds": 0.1}):
            with self.assertRaises(ValueError):
                Config.from_dict(values)

    def test_gaussian_underflow_does_not_create_zero_or_nan_scores(self):
        frequencies = np.linspace(0.5, 4, 8)
        with self.assertRaises(ValueError):
            gaussian_weights(frequencies, 100, 1)
        normalized = gaussian_weights(frequencies, 100, 1, "normalized")
        self.assertTrue(np.isfinite(normalized).all())
        self.assertAlmostEqual(float(normalized.sum()), 1)

    def test_invalid_path_types_are_rejected(self):
        for value in (None, True, 123, b"data/file", " ", "bad\x00path"):
            with self.assertRaises(ValueError):
                relative_path(value)

    def test_duplicate_and_ragged_csv_rows_are_rejected(self):
        with TemporaryWorkspace() as root:
            path = root / "manifest.csv"
            for text in ("label,label\n0,1\n", "video,label\ndata/file\n", "video,label\ndata/file,0,extra\n"):
                path.write_text(text, encoding="ascii")
                with self.assertRaises(ValueError):
                    read_csv(path)

    def test_failed_csv_write_preserves_existing_manifest(self):
        with TemporaryWorkspace() as root:
            path = root / "manifest.csv"
            path.write_text("original", encoding="ascii")
            with patch("support.storage.csv.DictWriter.writerows", side_effect=ValueError("Invalid row")):
                with self.assertRaises(ValueError):
                    write_csv(path, [{"label": 0}], ["label"])
            self.assertEqual(path.read_text(encoding="ascii"), "original")
            self.assertEqual(len(list(root.iterdir())), 1)

    def test_failed_npz_write_preserves_existing_archive(self):
        with TemporaryWorkspace() as root:
            path = root / "features.npz"
            path.write_bytes(b"original")
            with patch("numpy.savez_compressed", side_effect=ValueError("Packing failed")):
                with self.assertRaises(ValueError):
                    numpy_save(path, video=np.ones(2))
            self.assertEqual(path.read_bytes(), b"original")
            self.assertEqual(len(list(root.iterdir())), 1)

    def test_object_feature_arrays_are_rejected(self):
        with TemporaryWorkspace() as root:
            with self.assertRaises(ValueError):
                numpy_save(root / "features.npz", video=np.asarray([{}], dtype=object))
            self.assertFalse(list(root.iterdir()))

    def test_invalid_threshold_state_does_not_mutate_live_state(self):
        state = DynamicThreshold(Config())
        before = state.state_dict()
        with TemporaryWorkspace() as root:
            path = root / "state.json"
            for values in ({"threshold": 0.1, "history": [0.8], "accepted_count": 0}, {"accepted_count": 0.5}, {"version": True}, {"history": "bad"}):
                atomic_json(path, {**before, **values})
                with self.assertRaises(ValueError):
                    state.load(path)
                self.assertEqual(state.state_dict(), before)

    def test_large_finite_scores_keep_threshold_finite(self):
        state = DynamicThreshold(Config())
        state.decide(1e308)
        state.decide(1e308)
        self.assertTrue(np.isfinite(state.threshold))

    def test_stream_overflow_does_not_keep_rejected_packets(self):
        config = replace(Config(), image_size=32, maximum_clip_seconds=2)
        session = StreamSession(SimpleNamespace(config=config))
        frame = np.zeros((32, 32), dtype=np.uint8)
        session.push_video(frame, 0)
        with self.assertRaises(BufferError):
            session.push_video(frame, 3)
        self.assertEqual(len(session.frames), 1)
        self.assertEqual(session.last_video_time, 0)
        with self.assertRaises(BufferError):
            session.push_audio(np.ones(32001, dtype=np.float32), 0)
        self.assertEqual(len(session.audio), 0)
        self.assertIsNone(session.last_audio_end)

    def test_audio_packet_jitter_is_snapped_to_sample_timeline(self):
        session = StreamSession(SimpleNamespace(config=Config()))
        session.push_audio(np.ones(8000, dtype=np.float32), 0)
        session.push_audio(np.ones(8000, dtype=np.float32), 0.5 + 1 / 16000)
        self.assertEqual(session.audio[1][0], 0.5)
        self.assertEqual(session.last_audio_end, 1)

    def test_invalid_image_packet_does_not_advance_stream(self):
        session = StreamSession(SimpleNamespace(config=Config()))
        with self.assertRaises(ValueError):
            session.push_video(np.full((32, 32), np.nan), 0)
        self.assertFalse(session.frames)
        self.assertIsNone(session.last_video_time)

    def test_stereo_or_mismatched_clip_is_rejected_before_network(self):
        for waveform in (np.ones((32000, 2)), np.ones(32000)):
            count = 60 if waveform.ndim == 2 else 59
            clip = MediaClip(np.zeros((60, 32, 32), dtype=np.uint8), np.arange(count) / 30, waveform, 0, 16000, [])
            with self.assertRaises(InsufficientEvidence):
                prepare_inputs(clip, replace(Config(), image_size=32))

    def test_delayed_wav_origin_uses_actual_shared_coverage(self):
        config = replace(Config(), image_size=32)
        with TemporaryWorkspace() as root:
            video, audio = synthetic_media(root, "origin", config)
            clip = read_media(video, config, audio, duration=2, audio_offset=0.3)
            self.assertAlmostEqual(clip.start, 0.3)
            self.assertAlmostEqual(clip.end, 2)
            _, _, vt, at = prepare_inputs(clip, config)
            self.assertLess(abs(vt[0] - at[0]), config.alignment_tolerance)

    def test_ffmpeg_failures_become_recoverable_media_errors(self):
        for error in (subprocess.CalledProcessError(1, ["ffmpeg"]), subprocess.TimeoutExpired(["ffmpeg"], 60)):
            with patch("core.media.shutil.which", return_value="ffmpeg"), patch("core.media.subprocess.run", side_effect=error):
                with self.assertRaises(ValueError):
                    read_ffmpeg_audio("data/video.mp4", Config(), 0, 2)

    def test_ffprobe_invalid_metadata_is_rejected(self):
        result = SimpleNamespace(stdout=json.dumps({"streams": [{"codec_type": "video", "start_time": "nan"}, {"codec_type": "audio"}]}))
        with patch("core.media.shutil.which", return_value="ffprobe"), patch("core.media.subprocess.run", return_value=result):
            with self.assertRaises(ValueError):
                probe_stream_origins("data/video.mp4")

    def test_duration_mismatch_does_not_schedule_unbounded_windows(self):
        config = replace(Config(), image_size=32)
        with TemporaryWorkspace() as root:
            video, audio = synthetic_media(root, "duration", config)
            result = evaluate_row(SimpleNamespace(config=config), self.row(video=video, audio=audio, duration=1e100))
            self.assertEqual(result["decision"], "insufficient_evidence")
            self.assertEqual(result["total_segments"], 0)

    def test_sync_bounds_are_rejected_before_loading_weights(self):
        for values in ({"tolerance_seconds": np.nan}, {"maximum_offset": 0}):
            with patch("engine.sync.Detector") as mocked:
                with self.assertRaises(ValueError):
                    evaluate_sync([], "weights/model.pt", **values)
                mocked.assert_not_called()

    def test_unknown_sync_baseline_counts_every_requested_offset(self):
        rows = validate_rows([self.row()])
        with patch("engine.sync.Detector", return_value=SimpleNamespace(config=Config())):
            report = evaluate_sync(rows, "weights/model.pt", offsets=[-0.3, 0, 0.3])
        self.assertEqual(report["attempted_pairs"], 3)
        self.assertEqual(report["scored_pairs"], 0)

    def test_controlled_shift_does_not_invent_zero_baseline_lag(self):
        rows = validate_rows([self.row(sync_label=0)])
        clip = MediaClip(np.zeros((60, 32, 32), dtype=np.uint8), np.arange(60) / 30, np.ones(32000), 0, 16000, [])
        measurement = {"estimated_offset_seconds": 0, "predicted_sync_label": 0, "synchronization_score": 0.9, "patent_decision": "real"}
        with patch("engine.sync.Detector", return_value=SimpleNamespace(config=Config())), patch("engine.sync.read_media", return_value=clip), patch("engine.sync.sync_measurement", side_effect=lambda *args: dict(measurement)):
            report = evaluate_sync(rows, "weights/model.pt", offsets=[0, 0.03, 0.3])
        self.assertEqual(report["offset_metrics"]["annotated_pairs"], 0)
        self.assertIsNone(report["offset_metrics"]["mean_absolute_error_seconds"])
        self.assertEqual([item["true_sync_label"] for item in report["outcomes"]], [0, "", 1])

    def test_export_cannot_overwrite_checkpoint_or_other_encoder(self):
        for video, audio in (("weights/model.pt", "weights/audio.ts"), ("weights/shared.ts", "weights/shared.ts")):
            with patch("model.weights.load_encoders") as mocked:
                with self.assertRaises(ValueError):
                    export_encoders("weights/model.pt", video, audio)
                mocked.assert_not_called()

    def test_checkpoint_metadata_requires_integer_steps_and_tensor_entries(self):
        with TemporaryWorkspace() as root:
            path = root / "model.pt"
            path.touch()
            base = {"version": 1, "config": Config().to_dict(), "epoch": 1, "training_steps": 1, "provenance": "test", "model": {"bad": "not_a_tensor"}}
            for values in ({}, {"training_steps": 1.5}, {"training_steps": True}, {"model": []}, {"config": []}):
                with patch("model.weights.torch.load", return_value={**base, **values}), self.assertRaises(ValueError):
                    read_checkpoint(path)

    def test_binary_metrics_do_not_truncate_fractional_labels(self):
        for labels, predictions in (([0.5, 1], [0, 1]), ([0, 1], [0, 0.5])):
            with self.assertRaises(ValueError):
                binary_metrics(labels, [0.1, 0.9], predictions)

    def test_configuration_file_requires_json_object(self):
        with TemporaryWorkspace() as root:
            path = root / "config.json"
            atomic_json(path, [])
            with self.assertRaises(ValueError):
                load_config(path)

    def test_audio_resampling_rejects_fractional_rates_and_empty_audio(self):
        for waveform, rate in ((np.ones(10), 44100.5), (np.ones(10), True), (np.array([]), 16000)):
            with self.assertRaises(ValueError):
                mono_resample(waveform, rate, 16000)

    def test_large_pts_training_times_keep_dense_resolution(self):
        config = replace(Config(), image_size=32)
        row = validate_rows([self.row(split="train")])[0]
        origin = 1e8
        clip = MediaClip(np.zeros((60, 32, 32), dtype=np.uint8), origin + np.arange(60) / 30, np.ones(32000), origin, 16000, [])
        dataset = RealSegmentDataset([row], config)
        with patch("dataset.samples.probe_duration", return_value=2), patch("dataset.samples.read_media", return_value=clip):
            sample = dataset[0]
        self.assertTrue(bool((sample["audio_times"].diff() > 0).all()))
        self.assertLess(float(sample["audio_times"].abs().max()), 3)

    def test_dense_video_packets_cannot_bypass_time_buffer_limit(self):
        session = StreamSession(SimpleNamespace(config=replace(Config(), maximum_clip_seconds=2, image_size=32)))
        frame = np.zeros((32, 32), dtype=np.uint8)
        for index in range(61):
            session.push_video(frame, index / 100000)
        with self.assertRaises(BufferError):
            session.push_video(frame, 61 / 100000)
        self.assertEqual(len(session.frames), 61)

    def test_tiny_audio_packets_cannot_bypass_time_buffer_limit(self):
        session = StreamSession(SimpleNamespace(config=replace(Config(), maximum_clip_seconds=2)))
        for index in range(201):
            session.push_audio(np.ones(1, dtype=np.float32), index / 16000)
        with self.assertRaises(BufferError):
            session.push_audio(np.ones(1, dtype=np.float32), 201 / 16000)
        self.assertEqual(len(session.audio), 201)

    def test_prediction_outputs_cannot_overwrite_inputs_or_each_other(self):
        parser = parser_for_commands()
        base = ["predict", "--video", "data/example.mp4", "--checkpoint", "weights/model.pt"]
        for tail in (["--output", "data/../data/example.mp4"], ["--output", "runs/result.json", "--state", "runs/result.json"]):
            with self.assertRaises(ValueError):
                validate_outputs(parser.parse_args(base + tail))

    def test_resume_without_optimizer_is_rejected_cleanly(self):
        rows = validate_rows([self.row(video="data/train.mp4", split="train"), self.row(video="data/val.mp4", group="bob", source="second", split="val")])
        with patch("engine.train.read_checkpoint", return_value=({"model": {}}, Config())):
            with self.assertRaisesRegex(ValueError, "optimizer"):
                train(rows, Config(), "weights/model.pt", resume="weights/resume.pt")

    def test_nonfinite_training_rate_is_rejected_before_optimization(self):
        rows = validate_rows([self.row(video="data/train.mp4", split="train"), self.row(video="data/val.mp4", group="bob", source="second", split="val")])
        with patch("engine.train.torch.optim.AdamW") as mocked:
            with self.assertRaises(ValueError):
                train(rows, Config(), "weights/model.pt", learning_rate=float("nan"))
            mocked.assert_not_called()

    def test_import_outputs_protect_identity_maps_and_filelists(self):
        parser = parser_for_commands()
        for option in ("--identity-map", "--filelist"):
            args = parser.parse_args(["import", "--dataset", "custom", "--metadata", "data/input.csv", option, "data/helper.csv", "--output", "data/helper.csv"])
            with self.assertRaises(ValueError):
                validate_outputs(args)

    def test_split_inplace_exception_only_applies_to_manifest(self):
        parser = parser_for_commands()
        base = ["split", "--manifest", "data/manifest.csv", "--identity-map", "data/map.csv"]
        validate_outputs(parser.parse_args(base + ["--output", "data/manifest.csv"]))
        with self.assertRaises(ValueError):
            validate_outputs(parser.parse_args(base + ["--output", "data/map.csv"]))

    def test_cli_outputs_reject_existing_hardlink_aliases(self):
        with TemporaryWorkspace() as root:
            first, second = root / "result.json", root / "state.json"
            first.write_bytes(b"original")
            os.link(first, second)
            args = parser_for_commands().parse_args(["predict", "--video", "data/video.mp4", "--checkpoint", "weights/model.pt", "--output", str(first), "--state", str(second)])
            with self.assertRaises(ValueError):
                validate_outputs(args)
            self.assertEqual(first.read_bytes(), b"original")

    def test_direct_export_rejects_existing_hardlink_aliases(self):
        with TemporaryWorkspace() as root:
            checkpoint, video, audio = (root / name for name in ("model.pt", "video.ts", "audio.ts"))
            checkpoint.write_bytes(b"checkpoint")
            os.link(checkpoint, video)
            with patch("model.weights.load_encoders") as mocked, self.assertRaises(ValueError):
                export_encoders(checkpoint, video, audio)
            mocked.assert_not_called()
            video.unlink()
            video.write_bytes(b"encoder")
            os.link(video, audio)
            with patch("model.weights.load_encoders") as mocked, self.assertRaises(ValueError):
                export_encoders(checkpoint, video, audio)
            mocked.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), b"checkpoint")
            self.assertEqual(video.read_bytes(), b"encoder")

    def test_resume_may_update_checkpoint_but_cannot_overwrite_config(self):
        parser = parser_for_commands()
        base = ["train", "--manifest", "data/manifest.csv", "--resume", "weights/model.pt", "--checkpoint", "weights/model.pt"]
        validate_outputs(parser.parse_args(base))
        with self.assertRaises(ValueError):
            validate_outputs(parser.parse_args(base + ["--config", "weights/model.pt"]))

    def test_overall_fake_intervals_include_all_modal_annotations(self):
        row = validate_rows([self.row(label=1, duration=4, fake_segments="[[0,1]]", audio_segments="[[0.5,1.5],[2,3]]", video_segments="[[2.5,2.8]]")])[0]
        self.assertEqual(parse_segments(row["fake_segments"]), [[0, 1.5], [2, 3]])
        self.assertEqual(window_label(row, 2.1, 2.4), 1)
        self.assertEqual(window_label(row, 1.6, 1.9), 0)
        self.assertEqual(validate_rows([row])[0], row)

    def test_duplicate_or_adjacent_intervals_have_one_canonical_union(self):
        row = validate_rows([self.row(label=1, fake_segments="[[0,1],[0,1],[1,2]]")])[0]
        self.assertEqual(parse_segments(row["fake_segments"]), [[0, 2]])
        self.assertEqual(window_label(row, 2, 3), 0)

    def test_invalid_clip_metadata_returns_serializable_insufficient_evidence(self):
        detector = object.__new__(Detector)
        detector.config = Config()
        detector.extractor = SimpleNamespace(extract=None)
        for origin, rate in ((np.nan, 16000), (0, 0), (0, True)):
            clip = MediaClip(np.zeros((60, 32, 32)), np.arange(60) / 30, np.ones(32000), origin, rate, [])
            with patch.object(detector.extractor, "extract") as mocked:
                result = detector.process_segment(clip)
            mocked.assert_not_called()
            self.assertEqual(result["decision"], "insufficient_evidence")
            self.assertIsNone(result["segment_start"])
            self.assertIsNone(result["segment_end"])
            json.dumps(result, allow_nan=False)

    def test_disjoint_clip_coverage_does_not_update_threshold(self):
        detector = object.__new__(Detector)
        detector.config = Config()
        detector.threshold = DynamicThreshold(detector.config)
        detector.extractor = SimpleNamespace(extract=None)
        before = detector.threshold.state_dict()
        clip = MediaClip(np.zeros((60, 32, 32)), np.arange(60) / 30, np.ones(32000), 3, 16000, [])
        with patch.object(detector.extractor, "extract") as mocked:
            result = detector.process_segment(clip)
        mocked.assert_not_called()
        self.assertEqual(result["decision"], "insufficient_evidence")
        self.assertEqual(detector.threshold.state_dict(), before)

    def test_complex_and_object_features_are_rejected_before_interpolation(self):
        times = np.arange(60) / 30
        for values in (times[:, None] + 1j, np.full((60, 1), "bad", dtype=object)):
            with self.assertRaises(AlignmentError):
                align_features(values, times[:, None], times, times, Config())
            with self.assertRaises(AlignmentError):
                interpolate_features(values, times, times)

    def test_complex_and_object_timestamps_are_rejected(self):
        for times in (np.arange(60) / 30 + 1j, np.asarray(["0", "1"], dtype=object)):
            with self.assertRaises(AlignmentError):
                validate_times(times)

    def test_pca_rejects_complex_features_without_discarding_imaginary_part(self):
        with self.assertRaises(InsufficientEvidence):
            first_component(np.arange(60)[:, None] + 1j)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
