# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

CATALOG = {
    "lrs2": {"priority": 1, "source": "https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html", "purpose": "Authentic speaking clips for dense synchronization pretraining", "adapter": "dataset/speech.py:import_speech", "split": "Pass each official file list with its explicit subset; official splits use broadcast dates", "caveat": "Speech authenticity does not supply measured lag labels; controlled shifts require an explicitly synchronized baseline"},
    "lrs3": {"priority": 2, "source": "https://arxiv.org/abs/1809.00496", "purpose": "Speaking clips and out-of-domain synchronization experiments", "adapter": "dataset/speech.py:import_speech", "split": "Import local pretrain/trainval/test folders separately; create validation groups from training sources only", "caveat": "The historical Oxford download page returned 404 on 2026-10-09; use an authorized local copy"},
    "voxceleb2": {"priority": 3, "source": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html", "purpose": "Subject-grouped audio-video speech pretraining and synchronization stress tests", "adapter": "dataset/speech.py:import_speech", "split": "Use subject identifiers and retain original-video families; pass namespaces when mixing related datasets", "caveat": "The official site no longer supplies video/audio downloads; the adapter needs an existing authorized local copy"},
    "fakeavceleb": {
        "priority": 1,
        "source": "https://github.com/DASH-Lab/FakeAVCeleb",
        "purpose": "Start with multimodal video-level real, audio-fake, visual-fake, and both-fake comparisons",
        "adapter": "dataset/adapter.py:import_fakeavceleb",
        "split": "No split is guessed; use connected identity/source groups and a fixed seed",
        "access": "Request access from the dataset authors; the adapter reads a local authorized copy",
        "caveat": "Some generated fakes are intentionally lip-synchronized and can evade synchronization-only decisions",
    },
    "lavdf": {
        "priority": 2,
        "source": "https://github.com/ControlNet/LAV-DF",
        "purpose": "Test short manipulated intervals and window-level localization",
        "adapter": "dataset/adapter.py:import_lavdf",
        "metadata": "metadata.min.json: file, original, duration, fake_periods, n_fakes, modify_audio, modify_video, split",
        "split": "Preserve official train/dev/test partitions; dev is mapped to val and leakage is audited",
        "caveat": "When identity fields are absent, provide source/group identity mapping for subject-disjoint custom splits",
    },
    "avdeepfake1m": {
        "priority": 3,
        "source": "https://github.com/ControlNet/AV-Deepfake1M",
        "purpose": "Large-scale local manipulation and cross-dataset generalization",
        "adapter": "dataset/adapter.py:import_avdeepfake",
        "metadata": "file, original, split, modify_type, fake_segments; optional duration and subject_id",
        "split": "Keep official partitions; labels for some challenge test subsets are withheld",
        "access": "Follow the authors' EULA and local storage instructions",
        "caveat": "Do not invent hidden test labels; run labeled validation or submit through the official challenge interface",
    },
    "dfdc": {
        "priority": 4,
        "source": "https://ai.meta.com/datasets/dfdc/",
        "purpose": "Optional video-level external evaluation after checking every file has usable speech audio",
        "adapter": "dataset/adapter.py:import_dfdc",
        "metadata": "A filename dictionary with REAL/FAKE labels and original source filenames",
        "split": "Use original-family groups; actor-disjoint evaluation additionally needs actor identifiers",
        "caveat": "Visual manipulation may retain synchronization; audio availability must be validated",
    },
    "faceforensics": {
        "priority": 0,
        "source": "https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md",
        "purpose": "Visual-only backbone experiments",
        "adapter": None,
        "caveat": "The official downloaded sequences omit audio and cannot directly run the full patent pipeline",
    },
    "custom": {
        "priority": 5,
        "purpose": "Private talking-head, meeting, compression, and network-jitter experiments",
        "adapter": "dataset/manifest.py:read_manifest",
        "columns": ["dataset", "video", "audio", "label", "group", "source", "split", "duration", "audio_offset", "fake_segments", "category", "audio_segments", "video_segments", "sync_label", "offset_seconds", "identity_namespace", "source_namespace"],
        "label": "0=real, 1=fake; audio may point to a separate WAV file",
        "paths": "All paths are relative to the command working directory",
        "group": "Subject id; multiple source/target identities use semicolon-separated ids",
        "source_id": "Shared original video id for the genuine sample and all its generated variants",
        "fake_segments": "A JSON array of [start_seconds, end_seconds] pairs; blank means video-level annotations only",
        "sync_label": "Optional independent synchronization annotation: 0=synchronized, 1=out of sync, blank=unknown",
        "offset_seconds": "Optional measured audio-content lag; positive means audio lags video",
        "audio_offset": "WAV timeline origin; this field does not synthesize a content delay",
        "namespaces": "Declare shared source/identity namespaces to prevent leakage across related datasets",
    },
}
EXPERIMENTS = (
    "Use real training pairs to pretrain the encoders, real validation videos to calibrate, and the held-out test split to report metrics.",
    "Keep all identities and all variants of one source video inside one split.",
    "Report results separately for audio-only, visual-only, and combined manipulations.",
    "For cross-dataset evaluation, freeze encoders and calibration from the source dataset before running the target test set.",
    "If adapting to the target domain, calibrate only on its held-out real validation subset and label that experiment separately.",
    "Audit official splits before use; do not silently repair an official benchmark and call the new split official.",
    "Use --no-update or the default frozen evaluation mode for a stable offline benchmark.",
    "Use --adaptive only for a separately reported streaming experiment; state resets between unrelated videos.",
    "Report metric coverage and insufficient-evidence samples alongside AUC, EER, F1, and confusion counts.",
    "Use the sync command for measured offsets or controlled shifts; do not copy authenticity labels into sync labels.",
    "Known asynchronous clips are excluded from real-pair pretraining and real-validation calibration.",
)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
