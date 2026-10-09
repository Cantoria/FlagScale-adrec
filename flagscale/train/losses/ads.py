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

"""Objective of the advertising models: FuxiCTR ``get_total_loss``."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def embedding_regularization(parameters: list[nn.Parameter], weight: float) -> Tensor | float:
    """``sum(weight / 2 * ||W||_2^2)`` over the embedding parameters."""

    if weight <= 0 or not parameters:
        return 0.0
    return sum((weight / 2.0) * p.pow(2).sum() for p in parameters)


def ctr_loss(model: nn.Module, batch: dict[str, Tensor], device: torch.device,
             embedding_parameters: list[nn.Parameter], embedding_regularizer: float) -> Tensor:
    batch = move_batch(batch, device)
    logits = model(batch)
    # BCE on sigmoid outputs as in FuxiCTR, computed in the stable logit form.
    loss = F.binary_cross_entropy_with_logits(logits, batch["label"].to(logits.dtype))
    return loss + embedding_regularization(embedding_parameters, embedding_regularizer)


__all__ = ["ctr_loss", "embedding_regularization", "move_batch"]
