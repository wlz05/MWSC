# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import sys
sys.dont_write_bytecode = True
import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
from config import Config
from support.storage import read_json, atomic_json, relative_path, paths_match
from notes import PROJECT

def print_json(value):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and int(os.environ.get("RANK", "0")) != 0:
        return
    print(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False))

def config_option(parser):
    parser.add_argument("--config", help="Relative JSON configuration or calibration output")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")

def load_config(path):
    if path is None:
        return None, None
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("Configuration file requires a JSON object")
    return Config.from_dict(data.get("config", data)), data.get("model_id")

def validate_outputs(args):
    protected = ("video", "audio", "features", "manifest", "config", "video_backbone", "audio_backbone", "identity_map", "filelist", "resume")
    if args.command != "fetch":
        protected += ("syncnet_weights",)
    if args.command != "train":
        protected += ("checkpoint",)
    outputs = ("checkpoint",) if args.command == "train" else ("syncnet_weights", "example_output") if args.command == "fetch" else ("output", "state", "video_output", "audio_output")
    inputs = [(name, relative_path(getattr(args, name))) for name in protected if getattr(args, name, None)]
    inputs.extend(("metadata", relative_path(value)) for value in getattr(args, "metadata", None) or [])
    targets = [(name, relative_path(getattr(args, name)).resolve()) for name in outputs if getattr(args, name, None)]
    seen = []
    for name, target in targets:
        if any(paths_match(target, previous) for previous in seen):
            raise ValueError("Output paths must be distinct")
        seen.append(target)
        for input_name, source in inputs:
            if not paths_match(source, target):
                continue
            if args.command == "split" and name == "output" and input_name == "manifest":
                continue
            if args.command == "train" and name == "checkpoint" and input_name == "resume":
                continue
            raise ValueError("Output path would overwrite an input file")

def make_detector(args):
    from engine.detect import Detector
    config, expected_id = load_config(args.config)
    detector = Detector(args.checkpoint, args.device, getattr(args, "stream_id", "default"), config)
    if expected_id is not None and expected_id != detector.model_id:
        raise ValueError("Calibration belongs to a different checkpoint")
    return detector

def media_options(parser):
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", help="Separate relative WAV path")
    parser.add_argument("--audio-offset", type=float, default=0.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--checkpoint", required=True)
    config_option(parser)

def expert_options(parser):
    parser.add_argument("--syncnet-weights", default="../runtime/mwsc/syncnet.model")
    parser.add_argument("--sync-batch-size", type=int, default=8)
    parser.add_argument("--search-frames", type=int, default=15)
    parser.add_argument("--sync-window", type=float, default=5.0)
    parser.add_argument("--minimum-confidence", type=float, default=3.0)
    parser.add_argument("--sync-tolerance", type=float, default=0.08)
    faces = parser.add_mutually_exclusive_group()
    faces.add_argument("--face-box", nargs=4, type=float, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    faces.add_argument("--face-cropped", action="store_true")

def expert_arguments(args):
    return {"weights": args.syncnet_weights, "device": args.device, "batch_size": args.sync_batch_size, "vshift": args.search_frames, "face_box": args.face_box, "face_cropped": args.face_cropped, "minimum_confidence": args.minimum_confidence, "tolerance": args.sync_tolerance}

def parser_for_commands():
    parser = argparse.ArgumentParser(prog="MWSC", description=PROJECT["title"], epilog="Run from code/remain/MWSC. All paths are relative. Use notes for formulas, limitations, and examples.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("notes", help="Print provenance, equations, and full workflow")
    commands.add_parser("doctor", help="Inspect dependencies, GPU, and source format")
    commands.add_parser("datasets", help="Print dataset recommendations and schema")
    commands.add_parser("demo", help="Run synthetic feature equations without trained weights")
    commands.add_parser("test", help="Run local unit and integration tests")
    fetcher = commands.add_parser("fetch", help="Download and validate the official pretrained SyncNet v2 model")
    fetcher.add_argument("--syncnet-weights", default="../runtime/mwsc/syncnet.model")
    fetcher.add_argument("--example-output", help="Optional relative destination for the official face-cropped demo")
    lipreader = commands.add_parser("lipsync", help="Score one video using pretrained SyncNet without an MWSC checkpoint")
    lipreader.add_argument("--video", required=True)
    lipreader.add_argument("--audio")
    lipreader.add_argument("--audio-offset", type=float, default=0.0)
    lipreader.add_argument("--start", type=float, default=0.0)
    lipreader.add_argument("--duration", type=float)
    lipreader.add_argument("--device", default="auto")
    lipreader.add_argument("--output")
    expert_options(lipreader)
    importer = commands.add_parser("import", help="Convert local dataset metadata to a shared CSV manifest")
    importer.add_argument("--dataset", required=True, choices=("fakeavceleb", "lavdf", "avdeepfake1m", "dfdc", "lrs2", "lrs3", "voxceleb2", "custom"))
    importer.add_argument("--root", default="data")
    importer.add_argument("--metadata", nargs="+")
    importer.add_argument("--filelist", help="Relative list path; listed video entries are relative to --root")
    importer.add_argument("--subset", choices=("train", "val", "test"), default="")
    importer.add_argument("--audio-root", help="Optional parallel root with corresponding WAV files")
    importer.add_argument("--identity-namespace", help="Shared identity namespace for related datasets")
    importer.add_argument("--source-namespace", help="Shared original-source namespace for related datasets")
    importer.add_argument("--identity-map", help="CSV with source and group columns")
    importer.add_argument("--output", required=True)
    splitter = commands.add_parser("split", help="Build disjoint identity/source groups or audit official splits")
    splitter.add_argument("--manifest", required=True)
    splitter.add_argument("--output", required=True)
    splitter.add_argument("--seed", type=int, default=42)
    splitter.add_argument("--ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15))
    splitter.add_argument("--identity-map")
    splitter.add_argument("--keep-official", action="store_true")
    splitter.add_argument("--regroup", action="store_true")
    auditor = commands.add_parser("audit", help="Check labels, files, groups, and leakage")
    auditor.add_argument("--manifest", required=True)
    auditor.add_argument("--check-files", action="store_true")
    auditor.add_argument("--decode", action="store_true", help="Check usable audio-video windows before training")
    auditor.add_argument("--limit", type=int)
    config_option(auditor)
    trainer = commands.add_parser("train", help="Pretrain dense encoders using only real training pairs")
    trainer.add_argument("--manifest", required=True)
    trainer.add_argument("--checkpoint", required=True)
    trainer.add_argument("--epochs", type=int, default=10)
    trainer.add_argument("--batch-size", type=int, default=2)
    trainer.add_argument("--learning-rate", type=float, default=0.0001)
    trainer.add_argument("--workers", type=int, default=0)
    trainer.add_argument("--seed", type=int, default=42)
    trainer.add_argument("--resume")
    trainer.add_argument("--video-backbone")
    trainer.add_argument("--audio-backbone")
    trainer.add_argument("--architecture", choices=("compact", "i3d_vggish"))
    trainer.add_argument("--image-size", type=int)
    trainer.add_argument("--maximum-steps", type=int, help="Optional bounded smoke run")
    config_option(trainer)
    evaluator = commands.add_parser("evaluate", help="Run frozen or adaptive labeled validation/test evaluation")
    evaluator.add_argument("--manifest", required=True)
    evaluator.add_argument("--checkpoint", required=True)
    evaluator.add_argument("--split", choices=("val", "test"), default="test")
    evaluator.add_argument("--adaptive", action="store_true")
    evaluator.add_argument("--limit", type=int)
    evaluator.add_argument("--output")
    config_option(evaluator)
    synchronizer = commands.add_parser("sync", help="Evaluate annotated or controlled audio-video synchronization offsets")
    synchronizer.add_argument("--manifest", required=True)
    synchronizer.add_argument("--checkpoint", required=True)
    synchronizer.add_argument("--split", choices=("val", "test"), default="test")
    synchronizer.add_argument("--offsets", nargs="+", type=float, help="Controlled audio-content offsets in seconds")
    synchronizer.add_argument("--tolerance", type=float, default=0.05)
    synchronizer.add_argument("--maximum-offset", type=float, default=0.5)
    synchronizer.add_argument("--limit", type=int)
    synchronizer.add_argument("--output")
    config_option(synchronizer)
    calibration = commands.add_parser("calibrate", help="Set a threshold from real validation videos only")
    calibration.add_argument("--manifest", required=True)
    calibration.add_argument("--checkpoint", required=True)
    calibration.add_argument("--false-positive-rate", type=float, default=0.05)
    calibration.add_argument("--output", required=True)
    calibration.add_argument("--device", default="auto")
    predictor = commands.add_parser("predict", help="Decode media and classify overlapping segments")
    media_options(predictor)
    predictor.add_argument("--stream-id", default="default")
    predictor.add_argument("--state", help="Optional threshold state, bound to the stream and model")
    predictor.add_argument("--no-update", action="store_true")
    predictor.add_argument("--output")
    predictor.add_argument("--lipsync", action="store_true", help="Attach pretrained SyncNet lip-sync scores")
    expert_options(predictor)
    extractor = commands.add_parser("extract", help="Save dense neural feature sequences for one segment")
    media_options(extractor)
    extractor.add_argument("--output", required=True)
    scorer = commands.add_parser("score", help="Run PCA, CWT, coherence, and threshold on cached features")
    scorer.add_argument("--features", required=True)
    scorer.add_argument("--stream-id", default="feature_interface")
    scorer.add_argument("--state")
    scorer.add_argument("--no-update", action="store_true")
    scorer.add_argument("--output")
    config_option(scorer)
    exporter = commands.add_parser("export", help="Export trained dense encoder modules to TorchScript")
    exporter.add_argument("--checkpoint", required=True)
    exporter.add_argument("--video-output", required=True)
    exporter.add_argument("--audio-output", required=True)
    return parser

def doctor():
    from check import inspect_source
    dependencies = {name: importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "torch", "cv2", "av")}
    gpu = {"available": False}
    if dependencies["torch"]:
        import torch
        gpu = {"available": torch.cuda.is_available(), "torch_version": str(torch.__version__)}
        if gpu["available"]:
            gpu["devices"] = [{"name": torch.cuda.get_device_properties(index).name, "memory_gb": torch.cuda.get_device_properties(index).total_memory / 1024 ** 3} for index in range(torch.cuda.device_count())]
    expert = {"weights_available": Path("../runtime/mwsc/syncnet.model").is_file(), "automatic_faces_available": False}
    if dependencies["cv2"]:
        import cv2
        expert["opencv_version"] = cv2.__version__
        expert["automatic_faces_available"] = bool(getattr(cv2, "CascadeClassifier", None) or getattr(getattr(cv2, "xobjdetect", None), "CascadeClassifier", None) or getattr(cv2, "xobjdetect_CascadeClassifier", None))
    return {"project": PROJECT, "python_version": sys.version.split()[0], "dependencies": dependencies, "ffmpeg_on_path": shutil.which("ffmpeg") is not None, "ffprobe_on_path": shutil.which("ffprobe") is not None, "gpu": gpu, "lip_sync": expert, "source": inspect_source(), "media_note": "Install PyAV, use ffmpeg with ffprobe, or supply a separate WAV with constant-rate video"}

def demo():
    import numpy as np
    from engine.detect import score_features
    from core.threshold import DynamicThreshold
    config = Config()
    vt = np.arange(60) / 30
    at = np.arange(200) / 100
    def signal(times):
        return np.sin(2 * np.pi * (1.4 * times + 0.7 * times ** 2)) + 0.3 * np.sin(2 * np.pi * 3.7 * times)
    video = np.column_stack((signal(vt), 2 * signal(vt), -signal(vt)))
    audio = np.column_stack((signal(at), 1.5 * signal(at)))
    rng = np.random.default_rng(42)
    unrelated = rng.normal(size=(len(at), 2))
    matched = score_features(video, audio, vt, at, config, DynamicThreshold(config), False)
    mismatched = score_features(video, unrelated, vt, at, config, DynamicThreshold(config), False)
    return {"synthetic_only": True, "trained_detector_accuracy": "not evaluated", "matched_features": matched, "unrelated_features": mismatched}

def run(args):
    if args.command == "fetch":
        from support.download import fetch_assets
        return fetch_assets(args.syncnet_weights, args.example_output)
    if args.command == "lipsync":
        from engine.lipsync import LipSyncScorer
        scorer = LipSyncScorer(**expert_arguments(args))
        return scorer.predict(args.video, args.audio, args.start, args.duration, args.audio_offset, args.sync_window, args.output)
    if args.command == "notes":
        from notes import report
        return report()
    if args.command == "doctor":
        return doctor()
    if args.command == "datasets":
        from catalog import CATALOG, EXPERIMENTS
        return {"datasets": CATALOG, "experiments": EXPERIMENTS}
    if args.command == "demo":
        return demo()
    if args.command == "test":
        import unittest
        modules = ("verify.partition", "verify.system", "verify.formula", "verify.accuracy", "verify.process", "verify.regression", "verify.expert", "verify.parallel", "verify.protocol", "verify.format")
        suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        return {"tests_run": result.testsRun, "passed": True}
    if args.command == "import":
        from dataset.adapter import ADAPTERS, apply_identity_map
        from dataset.manifest import read_manifest, save_manifest, summary
        if args.dataset in ("lrs2", "lrs3", "voxceleb2"):
            from dataset.speech import import_speech
            if args.metadata:
                raise ValueError("Speech imports use --filelist; annotated CSV imports use --dataset custom")
            rows = import_speech(args.root, args.dataset, args.filelist, args.subset, args.audio_root)
        elif args.dataset == "fakeavceleb":
            rows = ADAPTERS[args.dataset](args.root)
        else:
            if not args.metadata:
                raise ValueError("This adapter requires --metadata")
            rows = []
            for path in args.metadata:
                rows.extend(read_manifest(path) if args.dataset == "custom" else ADAPTERS[args.dataset](args.root, path))
        if args.identity_map:
            rows = apply_identity_map(rows, args.identity_map)
        if args.identity_namespace:
            rows = [{**row, "identity_namespace": args.identity_namespace} for row in rows]
        if args.source_namespace:
            rows = [{**row, "source_namespace": args.source_namespace} for row in rows]
        save_manifest(args.output, rows)
        return {"output": args.output, **summary(rows)}
    if args.command in ("split", "audit"):
        from dataset.manifest import read_manifest, save_manifest
        from dataset.split import prepare_splits, audit_splits
        rows = read_manifest(args.manifest, getattr(args, "check_files", False))
        if args.command == "split":
            if args.identity_map:
                from dataset.adapter import apply_identity_map
                rows = apply_identity_map(rows, args.identity_map)
            rows = prepare_splits(rows, args.ratios, args.seed, args.keep_official, args.regroup)
            result = audit_splits(rows)
            save_manifest(args.output, rows)
            return {"output": args.output, **result}
        report = audit_splits(rows)
        if args.decode:
            from dataset.speech import audit_media
            config, _ = load_config(args.config)
            report["media_audit"] = audit_media(rows, config or Config(), args.limit)
        return report
    if args.command == "train":
        from dataset.manifest import read_manifest
        from engine.train import train
        config, _ = load_config(args.config)
        if args.resume and config is None:
            from model.weights import read_checkpoint
            _, config = read_checkpoint(args.resume)
        config = config or Config()
        if args.architecture:
            config = replace(config, architecture=args.architecture)
        if args.image_size:
            config = replace(config, image_size=args.image_size)
        return train(read_manifest(args.manifest, True, True), config, args.checkpoint, args.epochs, args.batch_size, args.learning_rate, args.workers, args.device, args.seed, args.resume, args.video_backbone, args.audio_backbone, args.maximum_steps)
    if args.command in ("evaluate", "calibrate", "sync"):
        from dataset.manifest import read_manifest
        from engine.evaluate import evaluate, calibrate
        rows = read_manifest(args.manifest, True, True)
        if args.command == "calibrate":
            return calibrate(rows, args.checkpoint, args.false_positive_rate, args.device, args.output)
        config, expected_id = load_config(args.config)
        if expected_id is not None:
            from engine.detect import checkpoint_identity
            if expected_id != checkpoint_identity(args.checkpoint):
                raise ValueError("Calibration belongs to a different checkpoint")
        if args.config and expected_id is not None:
            from dataset.provenance import verify_calibration
            verify_calibration(rows, read_json(args.config).get('validation_rows'), args.split)
        if args.command == "sync":
            from engine.sync import evaluate_sync
            return evaluate_sync(rows, args.checkpoint, args.split, args.device, config, args.offsets, args.tolerance, args.maximum_offset, args.limit, args.output)
        return evaluate(rows, args.checkpoint, args.split, args.device, config, args.adaptive, args.limit, args.output)
    if args.command in ("predict", "extract"):
        detector = make_detector(args)
        if args.command == "predict":
            if args.state and Path(args.state).exists():
                detector.threshold.load(args.state)
            result = detector.predict(args.video, args.audio, args.start, args.duration, args.audio_offset, not args.no_update)
            if args.lipsync:
                from engine.lipsync import add_lipsync
                result = add_lipsync(result, args.video, args.audio, **expert_arguments(args), start=args.start, duration=args.duration, audio_offset=args.audio_offset, window_seconds=args.sync_window)
            if args.state and not args.no_update:
                detector.threshold.save(args.state)
            if args.output:
                atomic_json(args.output, result)
            return result
        from core.media import read_media
        from engine.detect import save_features
        duration = args.duration if args.duration is not None else detector.config.segment_seconds
        if duration > detector.config.segment_seconds + 0.001:
            raise ValueError("Feature extraction exports one short segment at a time")
        clip = read_media(args.video, detector.config, args.audio, args.start, duration, args.audio_offset)
        features = detector.extractor.extract(clip)
        save_features(args.output, features, detector.config, detector.model_id)
        return {"output": args.output, "video_shape": list(features[0].shape), "audio_shape": list(features[1].shape), "model_id": detector.model_id}
    if args.command == "score":
        from engine.detect import load_features, score_features
        from core.threshold import DynamicThreshold
        features, config, model_id = load_features(args.features)
        override, expected_id = load_config(args.config)
        if expected_id is not None and expected_id != model_id:
            raise ValueError("Feature archive and calibration use different models")
        if override:
            protected = ("sample_rate", "video_fps", "preemphasis", "frame_ms", "hop_ms", "n_fft", "architecture", "feature_dim", "image_size")
            if any(getattr(override, key) != getattr(config, key) for key in protected):
                raise ValueError("Cached feature preprocessing cannot be changed")
            config = override
        threshold = DynamicThreshold(config, args.stream_id, model_id)
        if args.state and Path(args.state).exists():
            threshold.load(args.state)
        result = score_features(*features, config, threshold, not args.no_update)
        if args.state and not args.no_update:
            threshold.save(args.state)
        if args.output:
            atomic_json(args.output, result)
        return result
    if args.command == "export":
        from model.weights import export_encoders
        return export_encoders(args.checkpoint, args.video_output, args.audio_output)
    raise ValueError("Unknown command")

def main():
    parser = parser_for_commands()
    args = parser.parse_args()
    if not Path("config.py").is_file() or not Path("core/wavelet.py").is_file():
        parser.error("Change into the folder containing main.py before running commands")
    try:
        for key in ("config", "video", "audio", "checkpoint", "manifest", "root", "identity_map", "output", "state", "features", "resume", "video_backbone", "audio_backbone", "video_output", "audio_output", "filelist", "audio_root", "syncnet_weights", "example_output"):
            value = getattr(args, key, None)
            if value is not None:
                relative_path(value)
        for value in getattr(args, "metadata", None) or []:
            relative_path(value)
        validate_outputs(args)
        print_json(run(args))
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.exit(2, "Error: " + str(error) + "\n")

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
