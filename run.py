# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from support.storage import relative_path, paths_match

SEEDS = (42, 52, 3047, 3407)

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
    return parser

def main():
    parser = parser_for_runs()
    args = parser.parse_args()
    try:
        if not Path("main.py").is_file() or not Path("engine/train.py").is_file():
            raise ValueError("Run from the MWSC repository root")
        commands = build_commands(args)
        if args.dry_run:
            print(json.dumps({"seeds": args.seeds, "devices": args.devices, "commands": [command[1:] for command in commands]}, indent=2))
            return
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        # Run each seed on all selected devices.
        for seed, command in zip(args.seeds, commands):
            print("MWSC seed=" + str(seed), flush=True)
            subprocess.run(command, check=True, env=environment)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, "Error: " + str(error) + "\n")

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
