# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import numpy as np

def binary_metrics(labels, fake_scores, predictions):
    labels = np.asarray(labels)
    scores = np.asarray(fake_scores, dtype=np.float64)
    predictions = np.asarray(predictions)
    if labels.ndim != 1 or not len(labels) or labels.shape != scores.shape or labels.shape != predictions.shape:
        raise ValueError("Invalid metric arrays")
    if not np.isin(labels, (0, 1)).all() or not np.isin(predictions, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("Metrics require finite scores and binary labels")
    labels, predictions = labels.astype(int), predictions.astype(int)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {"count": len(labels), "accuracy": (tp + tn) / len(labels), "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "roc_auc": None, "eer": None}
    positives, negatives = int(labels.sum()), int(len(labels) - labels.sum())
    if positives and negatives:
        order = np.argsort(-scores, kind="stable")
        ordered_labels, ordered_scores = labels[order], scores[order]
        endpoints = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(labels) - 1]
        true_positive = np.r_[0, np.cumsum(ordered_labels)[endpoints]] / positives
        false_positive = np.r_[0, (endpoints + 1) - np.cumsum(ordered_labels)[endpoints]] / negatives
        result["roc_auc"] = float(np.sum(np.diff(false_positive) * (true_positive[1:] + true_positive[:-1]) / 2))
        false_negative = 1 - true_positive
        delta = false_positive - false_negative
        crossing = int(np.flatnonzero(delta >= 0)[0])
        if crossing == 0:
            eer = false_positive[0]
        else:
            weight = -delta[crossing - 1] / (delta[crossing] - delta[crossing - 1])
            eer = false_positive[crossing - 1] + weight * (false_positive[crossing] - false_positive[crossing - 1])
        result["eer"] = float(eer)
    return result

def interval_iou(left, right):
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0

def localization_metrics(predictions, ground_truth, thresholds=(0.3, 0.5, 0.7)):
    total_true = sum(len(intervals) for intervals in ground_truth.values())
    ranked = sorted(predictions, key=lambda item: item[3], reverse=True)
    result = {}
    for threshold in thresholds:
        used = {key: set() for key in ground_truth}
        true_positive, false_positive = [], []
        for key, start, end, score in ranked:
            intervals = ground_truth.get(key, [])
            candidates = [(interval_iou((start, end), interval), index) for index, interval in enumerate(intervals) if index not in used.get(key, set())]
            best = max(candidates, default=(0.0, -1))
            matched = best[0] >= threshold
            if matched:
                used[key].add(best[1])
            true_positive.append(int(matched))
            false_positive.append(int(not matched))
        if total_true == 0:
            result[str(threshold)] = {"average_precision": None, "recall": None, "ground_truth_intervals": 0}
            continue
        if not ranked:
            result[str(threshold)] = {"average_precision": 0.0, "recall": 0.0, "ground_truth_intervals": total_true}
            continue
        tp, fp = np.cumsum(true_positive), np.cumsum(false_positive)
        recall = tp / total_true
        precision = tp / np.maximum(tp + fp, 1)
        padded_recall = np.r_[0, recall, 1]
        padded_precision = np.r_[0, precision, 0]
        padded_precision = np.maximum.accumulate(padded_precision[::-1])[::-1]
        changes = np.flatnonzero(np.diff(padded_recall)) + 1
        ap = np.sum((padded_recall[changes] - padded_recall[changes - 1]) * padded_precision[changes])
        result[str(threshold)] = {"average_precision": float(ap), "recall": float(recall[-1]), "ground_truth_intervals": total_true}
    return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
