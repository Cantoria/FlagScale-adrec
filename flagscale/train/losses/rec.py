# Copyright 2026 FlagOS Contributors
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

"""Training objectives for the recommendation models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def sequential_loss(model: nn.Module, batch: dict[str, Tensor]) -> Tensor:
    """SASRec: BCE on each non-padded position's positive and negative.

    The reference indexes the valid positions with ``np.where``; a
    mask-weighted mean gives the same value with static shapes.
    """

    pos_logits, neg_logits = model(batch)
    mask = (batch["pos"] != 0).to(pos_logits.dtype)
    count = mask.sum().clamp_min(1.0)
    pos = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits), reduction="none")
    neg = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits), reduction="none")
    return ((pos + neg) * mask).sum() / count


def two_tower_loss(model: nn.Module, batch: dict[str, Tensor]) -> Tensor:
    """DSSM: point-wise BCE of ``sigmoid(<u, v>)`` against the 0/1 label."""

    logits = model(batch)
    return F.binary_cross_entropy_with_logits(logits, batch["label"].to(logits.dtype))


def bpr_loss(model: nn.Module, batch: dict[str, Tensor], weight_decay: float) -> Tensor:
    """LightGCN: ``mean(softplus(neg - pos)) + weight_decay * reg``."""

    out = model(batch)
    return F.softplus(out["neg_scores"] - out["pos_scores"]).mean() + weight_decay * out["reg"]


def loss_for_batch(model: nn.Module, task: str, batch: dict[str, Tensor], device: torch.device,
                   weight_decay: float = 0.0) -> Tensor:
    batch = move_batch(batch, device)
    if task == "sequential":
        return sequential_loss(model, batch)
    if task == "two_tower":
        return two_tower_loss(model, batch)
    if task == "graph":
        return bpr_loss(model, batch, weight_decay)
    raise ValueError(f"Unsupported rec task {task!r}")


__all__ = ["bpr_loss", "loss_for_batch", "move_batch", "sequential_loss", "two_tower_loss"]
