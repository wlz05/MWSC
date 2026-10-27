# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import random
import math
from itertools import chain
import numpy as np
import torch
from torch.utils.data import DataLoader
from support.storage import relative_path, protect_output
from dataset.split import audit_splits
from model.weights import read_checkpoint, save_checkpoint
from model.encoder import EncoderPair, device_from_name
from dataset.samples import RealSegmentDataset, collate_segments, aligned_batch
from engine.loss import temporal_contrastive_loss
from engine.parallel import ParallelContext

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_backbone(path):
    state = torch.load(relative_path(path, True), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Backbone requires a tensor state dictionary")
    return state

def run_epoch(model, loader, device, optimizer=None, maximum_steps=None, parallel=None):
    training = optimizer is not None
    model.train(training)
    total, count, examples = 0.0, 0, 0
    for batch in loader:
        if maximum_steps is not None and count >= maximum_steps:
            break
        with torch.set_grad_enabled(training):
            video, audio = model(batch["video"].to(device), batch["audio"].to(device))
            video, audio = aligned_batch(video, audio, batch["samples"])
            loss, _ = temporal_contrastive_loss(video, audio)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is nonfinite")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Training gradient is nonfinite")
                optimizer.step()
        batch_size = len(batch["samples"])
        total += float(loss.detach()) * batch_size
        examples += batch_size
        count += 1
    if parallel is not None:
        return parallel.weighted_mean(total, examples), count
    if examples == 0:
        raise ValueError("No batches were processed")
    return total / examples, count

def train(rows, config, checkpoint, epochs=10, batch_size=2, learning_rate=0.0001, workers=0, device="auto", seed=42, resume=None, video_backbone=None, audio_backbone=None, maximum_steps=None):
    audit = audit_splits(rows)
    if any(type(value) is not int for value in (epochs, batch_size, workers, seed)):
        raise ValueError("Epochs, batch size, workers, and seed require integers")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not math.isfinite(learning_rate):
        raise ValueError("Learning rate must be finite")
    if maximum_steps is not None and (type(maximum_steps) is not int or maximum_steps < 1):
        raise ValueError("Maximum steps must be a positive integer")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or workers < 0 or not 0 <= seed < 2 ** 32:
        raise ValueError("Invalid training parameters")
    # Resume may replace its checkpoint, while source media stay protected.
    protect_output(relative_path(checkpoint), chain((video_backbone, audio_backbone), (row.get(key) for row in rows for key in ("video", "audio"))))
    seed_everything(seed)
    config.validate()
    parallel = ParallelContext(device)
    try:
        return train_parallel(rows, config, checkpoint, epochs, batch_size, learning_rate, workers, seed, resume, video_backbone, audio_backbone, maximum_steps, audit, parallel)
    finally:
        parallel.close()

def train_parallel(rows, config, checkpoint, epochs, batch_size, learning_rate, workers, seed, resume, video_backbone, audio_backbone, maximum_steps, audit, parallel):
    target = parallel.device
    model = EncoderPair(config).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    start_epoch, steps = 0, 0
    provenance = "local_temporal_contrastive_pretraining"
    if resume:
        payload, saved_config = read_checkpoint(resume)
        if saved_config.signature() != config.signature():
            raise ValueError("Resume configuration differs from checkpoint")
        if not isinstance(payload.get("optimizer"), dict):
            raise ValueError("Resume requires an optimizer state")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"])
        steps = int(payload["training_steps"])
        provenance = payload["provenance"]
    elif video_backbone or audio_backbone:
        if config.architecture != "i3d_vggish" or not video_backbone or not audio_backbone:
            raise ValueError("Both backbone paths require the i3d_vggish architecture")
        model.video.import_backbone(load_backbone(video_backbone))
        model.audio.import_backbone(load_backbone(audio_backbone))
        provenance = "imported_backbones_and_local_temporal_pretraining"
    train_set = RealSegmentDataset(rows, config, "train", seed, True)
    validation_set = RealSegmentDataset(rows, config, "val", seed, False)
    sampler = parallel.training_sampler(train_set, seed)
    validation_subset = parallel.validation_subset(validation_set)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=workers, collate_fn=collate_segments, generator=generator)
    validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_segments)
    distributed_model = parallel.wrap(model)
    history = []
    best_validation = float("inf")
    for epoch in range(start_epoch, start_epoch + epochs):
        train_set.set_epoch(epoch)
        if sampler is not None:
            # Change the distributed shuffle order for every epoch.
            sampler.set_epoch(epoch)
        loss, epoch_steps = run_epoch(distributed_model, train_loader, target, optimizer, maximum_steps, parallel)
        parallel.synchronize_buffers(model)
        validation, _ = run_epoch(model, validation_loader, target, None, maximum_steps, parallel)
        steps += epoch_steps
        record = {"epoch": epoch + 1, "training_loss": loss, "validation_loss": validation, "training_steps": steps}
        history.append(record)
        if parallel.primary:
            print(record, flush=True)
        if validation < best_validation:
            best_validation = validation
            if parallel.primary:
                # Save only the selected checkpoint from rank zero.
                training = {"seed": seed, "world_size": parallel.world_size, "per_device_batch_size": batch_size, "global_batch_size": batch_size * parallel.world_size, "validation_loss": validation}
                save_checkpoint(checkpoint, model, config, epoch + 1, steps, optimizer, provenance, training)
        parallel.barrier()
    return {"checkpoint": str(checkpoint), "best_validation_loss": best_validation, "epochs": history, "split_audit": audit, "seed": seed, "world_size": parallel.world_size, "backend": parallel.backend, "per_device_batch_size": batch_size, "global_batch_size": batch_size * parallel.world_size, "objective": "Real-only temporal contrastive pretraining with rank-local negatives; an engineering extension"}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
