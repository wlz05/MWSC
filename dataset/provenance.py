# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import hashlib
import json
from dataset.manifest import validate_rows
from dataset.split import connected_groups

def partitions(rows):
    rows = validate_rows(rows, require_split=True)
    return {split: sorted((row for row in rows if row['split'] == split), key=lambda row: row['video']) for split in ('train', 'val')}

def signature(rows):
    payload = json.dumps(partitions(rows), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def saved_partitions(metadata):
    training = metadata.get('training', {}) if isinstance(metadata, dict) else {}
    saved = training.get('partitions') if isinstance(training, dict) else None
    if not isinstance(saved, dict) or set(saved) != {'train', 'val'} or any(not isinstance(saved[key], list) or not saved[key] for key in saved):
        raise ValueError('Checkpoint requires recorded train and validation partitions for this operation')
    rows = [row for split in ('train', 'val') for row in saved[split]]
    normalized = partitions(rows)
    if normalized != saved or signature(rows) != training.get('partition_signature'):
        raise ValueError('Checkpoint partition record is invalid')
    return saved

def verify_resume(rows, metadata):
    saved = saved_partitions(metadata)
    if partitions(rows) != saved:
        raise ValueError('Resume requires the original train and validation annotations')

def verify_holdout(rows, metadata, split):
    saved = saved_partitions(metadata)
    exposed = saved['train'] + (saved['val'] if split == 'test' else [])
    selected = [row for row in validate_rows(rows, require_split=True) if row['split'] == split]
    combined = exposed + selected
    boundary = len(exposed)
    # Compare current holdouts against every partition used by the checkpoint.
    for group in connected_groups(combined):
        if any(index < boundary for index in group) and any(index >= boundary for index in group):
            raise ValueError('Holdout membership overlaps checkpoint training or selection data')

def verify_calibration(rows, record, split):
    if not isinstance(record, list) or not record:
        raise ValueError('Calibration requires recorded validation annotations')
    reference = validate_rows(record, require_split=True)
    if any(row['split'] != 'val' or row['label'] != 0 for row in reference):
        raise ValueError('Calibration annotations must be real validation rows')
    if split == 'test':
        combined = reference + [row for row in rows if row['split'] == split]
        for group in connected_groups(combined):
            if any(index < len(reference) for index in group) and any(index >= len(reference) for index in group):
                raise ValueError('Test membership overlaps threshold calibration data')

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
