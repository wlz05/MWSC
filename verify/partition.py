# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import json
import unittest
import os
from config import Config
from support.storage import atomic_json, relative_path
from dataset.adapter import import_avdeepfake, import_lavdf, import_fakeavceleb, import_dfdc, apply_identity_map
from dataset.manifest import validate_rows, save_manifest, read_manifest, window_label
from dataset.split import grouped_split, audit_splits, connected_groups
from verify.fixture import TemporaryWorkspace

def make_rows():
    rows = []
    for subject in range(15):
        for label in (0, 1):
            rows.append({"dataset": "custom", "video": "data/subject%d_%d.mp4" % (subject, label), "audio": "", "label": label, "group": "person%d" % subject, "source": "source%d" % subject, "split": "", "duration": 4, "fake_segments": "[[1,2]]" if label else "[]", "category": "fake" if label else "real"})
    return rows

class ManifestTests(unittest.TestCase):
    def test_manifest_roundtrip(self):
        with TemporaryWorkspace() as root:
            path = root / "manifest.csv"
            rows = validate_rows(make_rows())
            save_manifest(path, rows)
            self.assertEqual(read_manifest(path), rows)

    def test_absolute_input_is_rejected(self):
        with self.assertRaises(ValueError):
            relative_path(str(__import__("pathlib").Path.cwd()))

    def test_duplicate_video_is_rejected(self):
        rows = make_rows()
        with self.assertRaises(ValueError):
            validate_rows(rows + [rows[0]])

    def test_local_labels_respect_fake_intervals(self):
        row = make_rows()[1]
        self.assertEqual(window_label(row, 0, 1), 0)
        self.assertEqual(window_label(row, 0.5, 1.5), 1)
        self.assertEqual(window_label(row, 2, 3), 0)

    def test_real_interval_conflict_is_rejected(self):
        row = make_rows()[0]
        row["fake_segments"] = "[[1,2]]"
        with self.assertRaises(ValueError):
            validate_rows([row])

class SplitTests(unittest.TestCase):
    def test_shared_audio_links_different_dataset_namespaces(self):
        rows = make_rows()[:2]
        rows[1].update(dataset="other", group="another", source="another")
        for row in rows:
            row["audio"] = "data/shared.wav"
        self.assertEqual(len(connected_groups(rows)), 1)
        rows[0]["split"], rows[1]["split"] = "train", "test"
        with self.assertRaisesRegex(ValueError, "leakage"):
            audit_splits(rows)

    def test_hardlinked_media_cannot_cross_dataset_splits(self):
        with TemporaryWorkspace() as root:
            first, second = root / "first.avi", root / "second.avi"
            first.write_bytes(b"shared media")
            os.link(first, second)
            rows = make_rows()[:2]
            rows[0].update(video=str(first), split="train")
            rows[1].update(dataset="other", video=str(second), group="another", source="another", split="test")
            with self.assertRaisesRegex(ValueError, "leakage"):
                audit_splits(rows)

    def test_grouped_split_is_reproducible_and_disjoint(self):
        first = grouped_split(make_rows(), seed=123)
        second = grouped_split(make_rows(), seed=123)
        self.assertEqual(first, second)
        self.assertEqual({row["split"] for row in first}, {"train", "val", "test"})
        self.assertEqual(audit_splits(first)["connected_groups"], 15)
        for group in connected_groups(first):
            self.assertEqual(len({first[index]["split"] for index in group}), 1)

    def test_source_link_unites_different_identity_labels(self):
        rows = make_rows()[:2]
        rows[1]["group"] = "target_person"
        self.assertEqual(len(connected_groups(rows)), 1)

    def test_multiple_identities_form_transitive_components(self):
        rows = make_rows()[:4]
        for index, row in enumerate(rows):
            row["source"] = "source%d" % index
        rows[0]["group"] = "a;b"
        rows[1]["group"] = "b;c"
        rows[2]["group"] = "c;d"
        rows[3]["group"] = "separate"
        self.assertEqual(sorted(map(len, connected_groups(rows))), [1, 3])

    def test_leakage_is_detected(self):
        rows = grouped_split(make_rows())
        rows[1]["split"] = "test" if rows[0]["split"] != "test" else "train"
        with self.assertRaises(ValueError):
            audit_splits(rows)

    def test_too_few_groups_is_rejected(self):
        with self.assertRaises(ValueError):
            grouped_split(make_rows()[:2])

class AdapterTests(unittest.TestCase):
    def test_avdeepfake_metadata_preserves_modality_and_time(self):
        with TemporaryWorkspace() as root:
            path = root / "metadata.json"
            atomic_json(path, [{"file": "train/id0001/fake.mp4", "original": "train/id0001/real.mp4", "split": "train", "modify_type": "audio_modified", "fake_segments": [[0.5, 1.0]]}])
            rows = import_avdeepfake(str(root / "data"), path)
            self.assertEqual(rows[0]["label"], 1)
            self.assertEqual(rows[0]["group"], "id0001")
            self.assertEqual(json.loads(rows[0]["fake_segments"]), [[0.5, 1]])

    def test_lavdf_dev_maps_to_validation(self):
        with TemporaryWorkspace() as root:
            path = root / "metadata.json"
            atomic_json(path, [{"file": "real/one.mp4", "original": None, "split": "dev", "n_fakes": 0, "fake_periods": [], "modify_video": False, "modify_audio": False, "duration": 3}])
            row = import_lavdf(str(root / "data"), path)[0]
            self.assertEqual(row["split"], "val")
            self.assertEqual(row["label"], 0)
            self.assertTrue(row["group"].startswith("source:"))

    def test_fakeav_folder_categories(self):
        with TemporaryWorkspace() as root:
            for category in ("RealVideo-RealAudio", "FakeVideo-FakeAudio"):
                path = root / category / "id0001" / "0001.mp4"
                path.parent.mkdir(parents=True)
                path.touch()
            rows = import_fakeavceleb(root)
            self.assertEqual(sorted(row["label"] for row in rows), [0, 1])
            self.assertEqual(len(connected_groups(rows)), 1)

    def test_dfdc_original_family_is_retained(self):
        with TemporaryWorkspace() as root:
            path = root / "metadata.json"
            atomic_json(path, {"real.mp4": {"label": "REAL"}, "fake.mp4": {"label": "FAKE", "original": "real.mp4"}})
            rows = import_dfdc(str(root / "data"), path)
            self.assertEqual(len(connected_groups(rows)), 1)

    def test_identity_map_requires_complete_sources(self):
        with TemporaryWorkspace() as root:
            path = root / "identities.csv"
            path.write_text("source,group\nsource0,person0\n", encoding="ascii")
            with self.assertRaises(ValueError):
                apply_identity_map(make_rows(), path)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
