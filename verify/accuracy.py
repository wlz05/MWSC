# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import unittest
from support.metrics import binary_metrics, interval_iou, localization_metrics

class MetricTests(unittest.TestCase):
    def test_perfect_fake_positive_auc(self):
        result = binary_metrics([0, 0, 1, 1], [-0.9, -0.8, -0.3, -0.2], [0, 0, 1, 1])
        self.assertEqual(result["roc_auc"], 1)
        self.assertEqual(result["eer"], 0)
        self.assertEqual(result["f1"], 1)

    def test_tied_scores_have_half_auc(self):
        result = binary_metrics([0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0])
        self.assertEqual(result["roc_auc"], 0.5)
        self.assertEqual(result["eer"], 0.5)

    def test_one_class_auc_is_unavailable(self):
        self.assertIsNone(binary_metrics([0, 0], [0, 1], [0, 0])["roc_auc"])

    def test_interval_iou(self):
        self.assertAlmostEqual(interval_iou((0, 2), (1, 3)), 1 / 3)

    def test_duplicate_proposals_count_as_false_positives(self):
        truth = {"one": [[1, 2]], "two": [[3, 4]]}
        predictions = [("one", 1, 2, 0.9), ("one", 1, 2, 0.8), ("two", 3, 4, 0.7)]
        result = localization_metrics(predictions, truth)["0.5"]
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertEqual(result["recall"], 1)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
