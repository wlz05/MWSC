# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import torch
from config import Config
from dataset.manifest import validate_rows
from dataset.adapter import import_avdeepfake
from support.storage import atomic_json
from dataset.split import connected_groups, audit_splits, prepare_splits
from dataset.provenance import partitions, signature, verify_resume, verify_holdout, verify_calibration
from engine.train import train
from model.encoder import EncoderPair
from model.weights import save_checkpoint, read_checkpoint
from verify.fixture import TemporaryWorkspace

def rows():
    return validate_rows([{'dataset': 'custom', 'video': 'data/' + split + '.mp4', 'label': 0, 'group': split, 'source': split, 'split': split} for split in ('train', 'val', 'test')])

def metadata(values):
    return {'training': {'partitions': partitions(values), 'partition_signature': signature(values)}}

class ProtocolTests(unittest.TestCase):
    def test_avdeepfake_resolves_sdk_subset_paths_and_frame_duration(self):
        with TemporaryWorkspace() as root:
            path = root / 'train_metadata.json'
            atomic_json(path, [{'file': 'real/one.mp4', 'original': 'vox/id0001/one.mp4', 'split': 'train', 'modify_type': 'real', 'video_frames': 50}])
            imported = import_avdeepfake(str(root / 'data'), path)
            self.assertTrue(imported[0]['video'].endswith('/data/train/real/one.mp4'))
            self.assertEqual(imported[0]['duration'], 2.0)

    def test_supplied_partitions_are_preserved_by_default(self):
        values = rows()
        self.assertEqual(prepare_splits(values, seed=3047), values)
        self.assertEqual(prepare_splits(values, keep_official=True), values)

    def test_partial_supplied_partitions_require_completion(self):
        values = rows()
        values[0]['split'] = ''
        with self.assertRaisesRegex(ValueError, 'Complete'):
            prepare_splits(values)
        self.assertEqual({row['split'] for row in prepare_splits(values, regroup=True)}, {'train', 'val', 'test'})

    def test_conflicting_split_modes_are_rejected(self):
        with self.assertRaises(ValueError):
            prepare_splits(rows(), keep_official=True, regroup=True)

    def test_declared_original_links_a_released_real_video(self):
        values = rows()[:2]
        values[0].update(video='data/train/real.mp4', source='upstream/original.mp4')
        values[1].update(video='data/fake.mp4', source='train/real.mp4')
        self.assertEqual(len(connected_groups(values)), 1)
        with self.assertRaises(ValueError):
            audit_splits(values)

    def test_repeated_clip_basenames_do_not_join_unrelated_sources(self):
        values = rows()[:2]
        values[0].update(video='data/alice/00001.mp4', source='alice')
        values[1].update(video='data/bob/00001.mp4', source='bob')
        self.assertEqual(len(connected_groups(values)), 2)

    def test_resume_requires_identical_train_and_validation_annotations(self):
        values = rows()
        verify_resume(list(reversed(values)), metadata(values))
        changed = deepcopy(values)
        changed[0]['label'] = 1
        with self.assertRaisesRegex(ValueError, 'original'):
            verify_resume(changed, metadata(values))

    def test_resume_rejects_changed_membership_within_disjoint_folds(self):
        values = rows()
        changed = deepcopy(values)
        changed[0]['split'], changed[2]['split'] = 'test', 'train'
        audit_splits(changed)
        with self.assertRaises(ValueError):
            verify_resume(changed, metadata(values))

    def test_final_test_rejects_former_training_and_selection_rows(self):
        values = rows()
        for index in (0, 1):
            selected = [{**values[index], 'split': 'test'}]
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                verify_holdout(selected, metadata(values), 'test')

    def test_validation_accepts_selection_rows_and_rejects_training_rows(self):
        values = rows()
        verify_holdout(values, metadata(values), 'val')
        verify_holdout(values, metadata(values), 'test')
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            verify_holdout([{**values[0], 'split': 'val'}], metadata(values), 'val')

    def test_missing_or_tampered_checkpoint_partitions_are_rejected(self):
        values = rows()
        for saved in ({}, {'training': {}}, metadata(values)):
            if saved.get('training', {}).get('partitions'):
                saved['training']['partitions']['train'][0]['label'] = 1
            with self.assertRaises(ValueError):
                verify_holdout(values, saved, 'test')

    def test_calibration_membership_is_checked_for_final_testing(self):
        values = rows()
        reference = [values[1]]
        verify_calibration(values, reference, 'test')
        with self.assertRaises(ValueError):
            verify_calibration([{**values[1], 'split': 'test'}], reference, 'test')

    def test_resumed_checkpoint_copy_is_written_only_by_rank_zero(self):
        values = rows()
        config = replace(Config(), image_size=32, feature_dim=8)
        with TemporaryWorkspace() as root:
            model = EncoderPair(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
            saved = metadata(values)["training"]
            saved["validation_loss"] = 0.1
            original, target = root / "original.pt", root / "resumed.pt"
            save_checkpoint(original, model, config, 2, 3, optimizer, training=saved)
            for primary in (True, False):
                context = SimpleNamespace(device=torch.device("cpu"), world_size=2, primary=primary, backend="gloo", training_sampler=lambda *args: None, validation_subset=lambda dataset: dataset, wrap=lambda model: model, synchronize_buffers=Mock(), barrier=Mock(), close=Mock())
                with patch("engine.train.ParallelContext", return_value=context), patch("engine.train.run_epoch", side_effect=[(1.0, 1), (0.2, 1)]), patch("engine.train.save_checkpoint") as writer:
                    train(values, config, target, epochs=1, device="cpu", resume=original)
                self.assertEqual(writer.call_count, int(primary))
                self.assertEqual(context.barrier.call_count, 2)
                context.close.assert_called_once()

    def test_resume_preserves_a_better_selected_checkpoint_at_a_new_path(self):
        values = rows()
        config = replace(Config(), image_size=32, feature_dim=8)
        with TemporaryWorkspace() as root:
            model = EncoderPair(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
            saved = metadata(values)['training']
            saved['validation_loss'] = 0.1
            original, target = root / 'original.pt', root / 'resumed.pt'
            save_checkpoint(original, model, config, 2, 3, optimizer, training=saved)
            with patch('engine.train.run_epoch', side_effect=[(1.0, 1), (0.2, 1)]):
                result = train(values, config, target, epochs=1, device='cpu', resume=original)
            self.assertEqual(result['best_validation_loss'], 0.1)
            payload, _ = read_checkpoint(target)
            self.assertEqual(payload['epoch'], 2)
            self.assertEqual(payload['training']['validation_loss'], 0.1)

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
