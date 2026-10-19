# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

from datetime import timedelta
import os
import torch
from torch import distributed as dist
from torch.utils.data import DistributedSampler, Subset
from model.encoder import device_from_name

class ParallelContext:
    def __init__(self, device="auto"):
        try:
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.rank = int(os.environ.get("RANK", "0"))
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        except ValueError as error:
            raise ValueError("Distributed ranks require integer environment values") from error
        if self.world_size < 1 or not 0 <= self.rank < self.world_size or self.local_rank < 0:
            raise ValueError("Invalid distributed rank or world size")
        self.owned = False
        self.device = device_from_name(device)
        self.backend = "single"
        if self.world_size > 1:
            if not dist.is_available():
                raise RuntimeError("This PyTorch build does not support distributed training")
            if self.device.type == "cuda":
                if self.local_rank >= torch.cuda.device_count():
                    raise ValueError("Local rank exceeds the available CUDA devices")
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
                self.backend = "nccl"
                if not dist.is_nccl_available():
                    raise RuntimeError("CUDA distributed training requires a PyTorch NCCL build")
            else:
                self.backend = "gloo"
            if not dist.is_initialized():
                dist.init_process_group(self.backend, timeout=timedelta(minutes=5))
                self.owned = True
            elif dist.get_rank() != self.rank or dist.get_world_size() != self.world_size:
                raise ValueError("Existing process group differs from the launcher ranks")

    @property
    def primary(self):
        return self.rank == 0

    def wrap(self, model):
        if self.world_size == 1:
            return model
        options = {"device_ids": [self.local_rank], "output_device": self.local_rank} if self.device.type == "cuda" else {}
        return torch.nn.parallel.DistributedDataParallel(model, **options)

    def training_sampler(self, dataset, seed):
        if self.world_size == 1:
            return None
        if len(dataset) < self.world_size:
            raise ValueError("Distributed training requires at least one real sample per rank")
        # Drop incomplete groups to avoid duplicate training samples.
        return DistributedSampler(dataset, self.world_size, self.rank, shuffle=True, seed=seed, drop_last=True)

    def validation_subset(self, dataset):
        # Validation shards contain no repeated samples.
        return Subset(dataset, list(range(self.rank, len(dataset), self.world_size))) if self.world_size > 1 else dataset

    def weighted_mean(self, total, count):
        # Reduce loss sums and sample counts across ranks.
        values = torch.tensor([total, count], dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        if values[1].item() <= 0:
            raise ValueError("No batches were processed")
        return float((values[0] / values[1]).item())

    def synchronize_buffers(self, model):
        if self.world_size > 1:
            for value in model.buffers():
                dist.broadcast(value, src=0)

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def close(self):
        if self.owned and dist.is_initialized():
            dist.destroy_process_group()
            self.owned = False

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
