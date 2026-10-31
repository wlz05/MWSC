# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import argparse
import json
import math
import hashlib
import statistics
import os
from pathlib import Path
import subprocess
import sys
from support.storage import relative_path, paths_match, protect_output, atomic_json, read_json

SEEDS = (42, 52, 3047, 3407)

def seed_summary(records, seeds=SEEDS):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds):
        raise ValueError("Seeds must be distinct unsigned 32-bit integers")
    if any(not isinstance(row, dict) or type(row.get("seed")) is not int or "metrics" not in row or not isinstance(row.get("protocol"), dict) for row in records):
        raise ValueError("Every seed result needs metrics and an experiment protocol")
    if len(records) != len(seeds) or {row["seed"] for row in records} != set(seeds):
        raise ValueError("Missing or duplicate seed results; partial averages are forbidden")
    if any(row["protocol"] != records[0]["protocol"] for row in records):
        raise ValueError("All seeds must use the same inputs and experiment settings")
    def flatten(value, prefix=""):
        result = {}
        if isinstance(value, dict):
            for key, item in value.items():
                result.update(flatten(item, prefix + ("." if prefix else "") + str(key)))
        elif value is None or isinstance(value, (int, float)) and not isinstance(value, bool):
            if value is not None and not math.isfinite(value):
                raise ValueError("Seed metrics must be finite or explicitly undefined")
            result[prefix] = value
        return result
    byseed = {str(row["seed"]): row["metrics"] for row in records}
    flat = {seed: flatten(byseed[str(seed)]) for seed in seeds}
    metrics = {}
    for key in sorted(set().union(*(value.keys() for value in flat.values()))):
        values = [flat[seed].get(key) for seed in seeds]
        undefined = [seed for seed, value in zip(seeds, values) if value is None]
        metrics[key] = {"mean": statistics.mean(values) if not undefined else None, "sample_std": statistics.stdev(values) if not undefined and len(values) > 1 else None, "undefined_seeds": undefined}
    return {"seeds": seeds, "seed_count": len(seeds), "byseed": byseed, "metrics": metrics, "protocol": records[0]["protocol"], "aggregation": "Compute each seed's test metrics first, then take their equally weighted arithmetic mean."}

def evaluation_commands(args, seed):
    root = relative_path(args.checkpoint_root) / ("seed" + str(seed))
    checkpoint = root / "encoders.pt"
    calibration, report = root / "calibration.json", root / "test.json"
    for target in (calibration, report, root / "result.json", relative_path(args.checkpoint_root) / "summary.json"):
        protect_output(target, (args.manifest, args.config, args.video_backbone, args.audio_backbone, checkpoint))
    base = [sys.executable, "-B", "main.py"]
    options = ["--manifest", str(relative_path(args.manifest)), "--checkpoint", str(checkpoint), "--device", args.device]
    return [base + ["calibrate", *options, "--output", str(calibration)], base + ["evaluate", *options, "--config", str(calibration), "--split", "test", "--output", str(report)]]

def evaluate_seeds(args, environment, manifest_hash):
    from model.weights import read_checkpoint
    records = []
    for seed in args.seeds:
        root = relative_path(args.checkpoint_root) / ("seed" + str(seed))
        metadata, config = read_checkpoint(root / "encoders.pt")
        training = metadata.get("training")
        if not isinstance(training, dict) or type(training.get("seed")) is not int or training["seed"] != seed:
            raise ValueError("Checkpoint training seed differs from its declared run")
        if hashlib.sha256(relative_path(args.manifest, True).read_bytes()).hexdigest() != manifest_hash:
            raise ValueError("The manifest changed between training seeds")
        for command in evaluation_commands(args, seed):
            subprocess.run(command, check=True, env=environment)
        report = read_json(root / "test.json")
        if hashlib.sha256(relative_path(args.manifest, True).read_bytes()).hexdigest() != manifest_hash:
            raise ValueError("The manifest changed during evaluation")
        protocol = {"manifest_sha256": manifest_hash, "config_signature": config.signature(), "split": report["split"], "videos": sorted(row["video"] for row in report["predictions"])}
        record = {"seed": seed, "protocol": protocol, "metrics": {key: report[key] for key in ("clip_metrics", "window_metrics", "video_coverage", "requested_videos", "scored_videos")}}
        seed_summary([record], [seed])
        atomic_json(root / "result.json", record)
        records.append(record)
    result = seed_summary(records, args.seeds)
    atomic_json(relative_path(args.checkpoint_root) / "summary.json", result)
    return result

def build_commands(args):
    if type(args.devices) is not int or not 1 <= args.devices <= 8:
        raise ValueError("Devices must be an integer from 1 to 8")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in args.seeds):
        raise ValueError("Seeds must be distinct unsigned 32-bit integers")
    if any(type(value) is not int for value in (args.epochs, args.batch_size, args.workers)) or args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Epochs and batch size must be positive integers; workers must be nonnegative")
    if isinstance(args.learning_rate, bool) or not isinstance(args.learning_rate, (int, float)) or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if args.maximum_steps is not None and (type(args.maximum_steps) is not int or args.maximum_steps < 1):
        raise ValueError("Maximum steps must be a positive integer")
    if args.image_size is not None and (type(args.image_size) is not int or args.image_size < 32):
        raise ValueError("Image size must be an integer of at least 32")
    manifest = relative_path(args.manifest)
    root = relative_path(args.checkpoint_root)
    for value in (args.config, args.video_backbone, args.audio_backbone):
        if value is not None:
            relative_path(value)
    commands = []
    for seed in args.seeds:
        checkpoint = root / ("seed" + str(seed)) / "encoders.pt"
        if any(value is not None and paths_match(checkpoint, value) for value in (manifest, args.config, args.video_backbone, args.audio_backbone)):
            raise ValueError("Seed checkpoint cannot overwrite an input file")
        command = [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(args.devices), "main.py", "train", "--manifest", str(manifest), "--checkpoint", str(checkpoint), "--seed", str(seed), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--workers", str(args.workers), "--device", args.device, "--learning-rate", str(args.learning_rate)]
        for name in ("architecture", "image_size", "config", "video_backbone", "audio_backbone", "maximum_steps"):
            value = getattr(args, name)
            if value is not None:
                command.extend(("--" + name.replace("_", "-"), str(value)))
        commands.append(command)
    return commands

def parser_for_runs():
    parser = argparse.ArgumentParser(description="Run four MWSC seeds with one distributed process per GPU")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-root", default="../runtime/mwsc/experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--architecture", choices=("compact", "i3d_vggish"))
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--config")
    parser.add_argument("--video-backbone")
    parser.add_argument("--audio-backbone")
    parser.add_argument("--maximum-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument("--evaluate", action="store_true", help="Calibrate and test each seed, then average seed metrics")
    evaluation.add_argument("--evaluate-only", action="store_true", help="Evaluate existing seed checkpoints without retraining")
    return parser

def main():
    parser = parser_for_runs()
    args = parser.parse_args()
    try:
        if not Path("main.py").is_file() or not Path("engine/train.py").is_file():
            raise ValueError("Run from the MWSC repository root")
        commands = build_commands(args)
        plans = {str(seed): [command[1:] for command in evaluation_commands(args, seed)] for seed in args.seeds} if args.evaluate or args.evaluate_only else {}
        if args.dry_run:
            print(json.dumps({"seeds": args.seeds, "devices": args.devices, "commands": [] if args.evaluate_only else [command[1:] for command in commands], "evaluation": plans}, indent=2))
            return
        manifest_hash = hashlib.sha256(relative_path(args.manifest, True).read_bytes()).hexdigest()
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        # Run each seed on all selected devices.
        if not args.evaluate_only:
            for seed, command in zip(args.seeds, commands):
                if hashlib.sha256(relative_path(args.manifest, True).read_bytes()).hexdigest() != manifest_hash:
                    raise ValueError("The manifest changed between training seeds")
                print("MWSC seed=" + str(seed), flush=True)
                subprocess.run(command, check=True, env=environment)
        if args.evaluate or args.evaluate_only:
            print(json.dumps(evaluate_seeds(args, environment, manifest_hash), indent=2, allow_nan=False))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, "Error: " + str(error) + "\n")

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
