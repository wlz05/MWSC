# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from dataclasses import replace
import os
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import TensorDataset
from engine.parallel import ParallelContext
from engine.train import seed_everything
from run import SEEDS, build_commands, parser_for_runs

class ParallelTests(unittest.TestCase):
    def test_single_process_preserves_model_and_dataset(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}):
            context = ParallelContext("cpu")
        dataset = TensorDataset(torch.arange(5))
        model = torch.nn.Linear(2, 2)
        self.assertIs(context.wrap(model), model)
        self.assertIs(context.validation_subset(dataset), dataset)
        self.assertIsNone(context.training_sampler(dataset, 42))
        self.assertEqual(context.weighted_mean(12, 3), 4)
        context.close()

    def test_invalid_launcher_ranks_are_rejected(self):
        for values in ({"WORLD_SIZE": "bad"}, {"WORLD_SIZE": "0"}, {"WORLD_SIZE": "2", "RANK": "2"}, {"LOCAL_RANK": "-1"}):
            with patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0", **values}), self.assertRaises(ValueError):
                ParallelContext("cpu")

    def test_validation_shards_cover_samples_without_padding(self):
        dataset = TensorDataset(torch.arange(5))
        shards = []
        for rank in range(3):
            context = object.__new__(ParallelContext)
            context.world_size, context.rank = 3, rank
            shards.append(context.validation_subset(dataset).indices)
        self.assertEqual(sorted(sum(shards, [])), list(range(5)))
        self.assertFalse(set(shards[0]) & set(shards[1]))

    def test_training_sampler_drops_incomplete_groups_and_is_reproducible(self):
        dataset = TensorDataset(torch.arange(9))
        shards = []
        for rank in range(2):
            context = object.__new__(ParallelContext)
            context.world_size, context.rank = 2, rank
            sampler = context.training_sampler(dataset, 42)
            first = list(sampler)
            self.assertEqual(first, list(context.training_sampler(dataset, 42)))
            shards.append(first)
        self.assertEqual(len(shards[0]), 4)
        self.assertFalse(set(shards[0]) & set(shards[1]))

    def test_training_requires_a_real_sample_for_every_rank(self):
        context = object.__new__(ParallelContext)
        context.world_size, context.rank = 8, 0
        with self.assertRaises(ValueError):
            context.training_sampler(TensorDataset(torch.arange(4)), 42)

    def test_four_seed_launcher_builds_separate_eight_gpu_runs(self):
        args = parser_for_runs().parse_args(["--manifest", "manifests/split.csv", "--dry-run"])
        commands = build_commands(args)
        self.assertEqual(tuple(args.seeds), SEEDS)
        self.assertEqual(len(commands), 4)
        for seed, command in zip(SEEDS, commands):
            self.assertEqual(command[command.index("--nproc_per_node") + 1], "8")
            self.assertEqual(command[command.index("--seed") + 1], str(seed))
            self.assertIn("seed" + str(seed), command[command.index("--checkpoint") + 1])

    def test_launcher_rejects_invalid_training_values_before_spawning(self):
        parser = parser_for_runs()
        for tail in (["--epochs", "0"], ["--batch-size", "0"], ["--workers", "-1"], ["--learning-rate", "nan"], ["--learning-rate", "inf"], ["--maximum-steps", "0"], ["--image-size", "16"]):
            with self.subTest(arguments=tail), self.assertRaises(ValueError):
                build_commands(parser.parse_args(["--manifest", "data/split.csv", *tail]))

    def test_seed_launcher_rejects_duplicate_seeds_and_absolute_paths(self):
        parser = parser_for_runs()
        for tail in (["--seeds", "42", "42"], ["--devices", "0"], ["--seeds", "-1"], ["--checkpoint-root", "".join(("C", ":", "/", "weights"))]):
            with self.assertRaises(ValueError):
                build_commands(parser.parse_args(["--manifest", "manifests/split.csv", *tail]))

    def test_requested_seed_repeats_tensor_initialization(self):
        for seed in SEEDS:
            seed_everything(seed)
            first = torch.rand(4)
            seed_everything(seed)
            self.assertTrue(torch.equal(first, torch.rand(4)))

    def test_empty_global_epoch_is_rejected(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}):
            context = ParallelContext("cpu")
        with self.assertRaises(ValueError):
            context.weighted_mean(0, 0)

    def test_rank_zero_is_the_only_checkpoint_writer(self):
        context = object.__new__(ParallelContext)
        for rank, expected in ((0, True), (1, False)):
            context.rank = rank
            self.assertEqual(context.primary, expected)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
