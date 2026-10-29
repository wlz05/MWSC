# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from dataclasses import replace
import unittest
import numpy as np
import torch
from config import Config
from engine.detect import Detector, save_features, load_features, score_features
from core.media import read_media
from dataset.manifest import validate_rows
from model.encoder import EncoderPair
from model.weights import save_checkpoint, load_encoders, export_encoders
from model.vision import DenseI3D
from model.sound import DenseVGGish
from engine.train import train
from engine.evaluate import evaluate, calibrate
from engine.sync import evaluate_sync
from verify.fixture import TemporaryWorkspace, synthetic_media

class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_compact_encoders_preserve_dense_time(self):
        config = replace(Config(), feature_dim=16, image_size=32)
        model = EncoderPair(config).eval()
        with torch.no_grad():
            video, audio = model(torch.randn(2, 1, 60, 32, 32), torch.randn(2, 1, 257, 198))
        self.assertEqual(tuple(video.shape), (2, 60, 16))
        self.assertEqual(tuple(audio.shape), (2, 198, 16))

    def test_dense_preferred_networks_preserve_time(self):
        video = DenseI3D(16).eval()
        audio = DenseVGGish(16).eval()
        with torch.no_grad():
            left = video(torch.randn(1, 1, 6, 32, 32))
            right = audio(torch.randn(1, 1, 64, 20))
        self.assertEqual(tuple(left.shape), (1, 6, 16))
        self.assertEqual(tuple(right.shape), (1, 20, 16))

    def test_missing_backbone_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            DenseVGGish(16).import_backbone({})

    def test_untrained_checkpoint_cannot_be_saved(self):
        with TemporaryWorkspace() as root:
            config = replace(Config(), feature_dim=16)
            with self.assertRaises(ValueError):
                save_checkpoint(root / "untrained.pt", EncoderPair(config), config, 0, 0)

    def test_training_prediction_calibration_evaluation_and_export(self):
        with TemporaryWorkspace() as root:
            config = replace(Config(), image_size=32, feature_dim=16)
            rows = []
            for split in ("train", "val", "test"):
                for index in range(2):
                    name = split + str(index)
                    video, audio = synthetic_media(root, name, config, index * 0.3)
                    rows.append({"dataset": "synthetic", "video": video, "audio": audio, "label": 0, "group": name, "source": name, "split": split, "duration": 2, "fake_segments": "[]", "category": "real"})
            rows = validate_rows(rows)
            checkpoint = root / "encoders.pt"
            training = train(rows, config, checkpoint, epochs=1, batch_size=2, device="cpu", maximum_steps=1)
            self.assertGreater(training["epochs"][0]["training_steps"], 0)
            detector = Detector(checkpoint, "cpu")
            result = detector.predict(rows[-1]["video"], rows[-1]["audio"], update=False)
            self.assertEqual(result["valid_segments"], 1)
            self.assertIn(result["decision"], ("real", "fake"))
            clip = read_media(rows[-1]["video"], config, rows[-1]["audio"])
            features = detector.extractor.extract(clip)
            archive = root / "features.npz"
            save_features(archive, features, config, detector.model_id)
            restored, restored_config, model_id = load_features(archive)
            self.assertEqual(restored_config, config)
            self.assertEqual(model_id, detector.model_id)
            score = score_features(*restored, config)
            self.assertAlmostEqual(score["synchronization_score"], result["segments"][0]["synchronization_score"], places=6)
            calibration = calibrate(rows, checkpoint, device="cpu", output=root / "calibration.json")
            calibrated_config = Config.from_dict(calibration["config"])
            evaluation = evaluate(rows, checkpoint, "test", "cpu", calibrated_config, output=root / "evaluation.json")
            self.assertEqual(evaluation["scored_videos"], 2)
            self.assertEqual(evaluation["video_coverage"], 1)
            synchronization_rows = [{**row, "sync_label": 0, "offset_seconds": 0.0} for row in rows]
            synchronization = evaluate_sync(synchronization_rows, checkpoint, "test", "cpu", offsets=[-0.2, 0, 0.2], output=root / "sync.json")
            self.assertEqual(synchronization["attempted_pairs"], 6)
            self.assertEqual(synchronization["scored_pairs"], 6)
            self.assertTrue(all(item["deepfake_label"] == 0 for item in synchronization["outcomes"]))
            export_encoders(checkpoint, root / "video.ts", root / "audio.ts")
            scripted_video = torch.jit.load(str(root / "video.ts"))
            model, _, _ = load_encoders(checkpoint, "cpu")
            values = torch.randn(1, 1, 60, 32, 32)
            with torch.no_grad():
                torch.testing.assert_close(scripted_video(values), model.video(values))

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
