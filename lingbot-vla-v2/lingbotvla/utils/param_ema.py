# Copyright 2026 Robbyant Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EMA of the trainable parameters for FSDP2 training, kept on the CPU.

Each rank keeps an fp32 CPU copy of its local shards (6B params / 4 ranks: ~6 GB host memory per rank,
no GPU memory) and updates it every ``every`` optimizer steps with decay ** every. To save, the shadow
weights are swapped into the model one tensor at a time and the model is written with the same DCP
layout as the "model" directory, into ``<checkpoint>/ema``; the HF saver then exports it as
``ema_hf_ckpt``. Frozen parameters are not tracked: they are equal in the model and in the EMA.
"""

import os

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

from ..checkpoint.checkpointer import ModelState
from . import logging


logger = logging.get_logger(__name__)

EMA_DIR = "ema"


def _local(p: torch.Tensor) -> torch.Tensor:
    return p._local_tensor if hasattr(p, "_local_tensor") else p


class ParamEMA:
    def __init__(self, model: torch.nn.Module, decay: float, every: int = 10):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema decay must be in (0, 1), got {decay}")
        if every < 1:
            raise ValueError(f"ema every must be >= 1, got {every}")
        self.decay = decay
        self.every = every
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shadow = [_local(p).detach().to("cpu", torch.float32, copy=True) for p in self.params]
        n = sum(s.numel() for s in self.shadow)
        logger.info_rank0(f"EMA decay={decay} every={every}: {len(self.params)} tensors, {n / 1e9:.2f}B local params on CPU (rank 0)")

    @torch.no_grad()
    def update(self, step: int) -> None:
        """Call after optimizer.step() with the new global step."""
        if step % self.every:
            return
        w = 1.0 - self.decay**self.every
        for p, s in zip(self.params, self.shadow):
            s.lerp_(_local(p).detach().to("cpu", torch.float32), w)

    @torch.no_grad()
    def _swap(self) -> None:
        """Exchange model weights and shadow weights (calling it twice restores both)."""
        for p, s in zip(self.params, self.shadow):
            loc = _local(p)
            tmp = loc.detach().to("cpu", torch.float32, copy=True)
            loc.copy_(s.to(loc.device, loc.dtype))
            s.copy_(tmp)

    def save(self, model: torch.nn.Module, checkpoint_dir: str) -> None:
        """Collective: write the EMA weights to <checkpoint_dir>/ema (same layout as <checkpoint_dir>/model)."""
        self._swap()
        try:
            dcp.save(
                state_dict={"state": ModelState(model)},
                storage_writer=FileSystemWriter(
                    os.path.join(checkpoint_dir, EMA_DIR), thread_count=16, single_file_per_rank=True, sync_files=False
                ),
            )
        finally:
            self._swap()
        logger.info_rank0(f"Saved EMA weights to {os.path.join(checkpoint_dir, EMA_DIR)}")

    def load(self, model: torch.nn.Module, checkpoint_dir: str) -> bool:
        """Collective: restore the shadow from <checkpoint_dir>/ema; False (shadow unchanged) if it is missing."""
        ema_dir = os.path.join(checkpoint_dir, EMA_DIR)
        if not os.path.exists(os.path.join(ema_dir, ".metadata")):
            logger.info_rank0(f"No EMA weights in {checkpoint_dir}: EMA restarts from the loaded model")
            return False
        self._swap()  # model <- shadow, shadow <- model weights
        try:
            dcp.load(state_dict={"state": ModelState(model)}, storage_reader=FileSystemReader(ema_dir))
        finally:
            self._swap()  # model <- model weights, shadow <- loaded EMA
        logger.info_rank0(f"Loaded EMA weights from {ema_dir}")
        return True
