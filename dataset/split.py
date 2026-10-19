# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from collections import defaultdict
import numpy as np
from dataset.manifest import SPLITS, validate_rows, summary

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        self.parent[self.find(left)] = self.find(right)

def connected_groups(rows):
    links = UnionFind(len(rows))
    owners = {}
    for index, row in enumerate(rows):
        dataset = row["dataset"]
        keys = [("source", row.get("source_namespace") or dataset, row["source"])]
        keys.extend(("identity", row.get("identity_namespace") or dataset, identity.strip()) for identity in row["group"].split(";") if identity.strip())
        for key in keys:
            if key in owners:
                links.union(index, owners[key])
            else:
                owners[key] = index
    groups = defaultdict(list)
    for index in range(len(rows)):
        groups[links.find(index)].append(index)
    return list(groups.values())

def audit_splits(rows):
    rows = validate_rows(rows, require_split=True)
    groups = connected_groups(rows)
    violations = []
    for group in groups:
        splits = {rows[index]["split"] for index in group}
        if len(splits) > 1:
            violations.append({"splits": sorted(splits), "videos": [rows[index]["video"] for index in group[:5]]})
    if violations:
        raise ValueError("Identity or source leakage across splits: " + str(violations[:3]))
    report = summary(rows)
    report["connected_groups"] = len(groups)
    report["source_only_groups"] = sum(row["group"].startswith("source:") for row in rows)
    report["identity_note"] = "Source-only grouping requires an identity map for subject-disjoint evaluation"
    return report

def grouped_split(rows, ratios=(0.7, 0.15, 0.15), seed=42):
    rows = validate_rows(rows)
    ratios = np.asarray(ratios, dtype=np.float64)
    if ratios.shape != (3,) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1):
        raise ValueError("Three positive split ratios must sum to one")
    groups = connected_groups(rows)
    if len(groups) < 3:
        raise ValueError("At least three independent identity/source groups are required")
    strata = sorted({(row["dataset"], row["label"]) for row in rows})
    columns = {key: index for index, key in enumerate(strata)}
    totals = np.zeros(len(strata))
    for row in rows:
        totals[columns[(row["dataset"], row["label"])]] += 1
    targets = ratios[:, None] * totals[None]
    counts = np.zeros_like(targets)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    assigned_groups = np.zeros(3, dtype=int)
    output = [dict(row) for row in rows]
    for order, group in enumerate(groups):
        histogram = np.zeros(len(strata))
        for index in group:
            row = rows[index]
            histogram[columns[(row["dataset"], row["label"])]] += 1
        candidates = list(range(3))
        empty = np.flatnonzero(assigned_groups == 0).tolist()
        if len(groups) - order == len(empty):
            candidates = empty
        costs = []
        for split in candidates:
            proposed = counts.copy()
            proposed[split] += histogram
            costs.append(float(np.sum((proposed - targets) ** 2 / np.maximum(targets, 1))))
        choice = candidates[int(np.argmin(costs))]
        counts[choice] += histogram
        assigned_groups[choice] += 1
        for index in group:
            output[index]["split"] = SPLITS[choice]
    audit_splits(output)
    return output

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
