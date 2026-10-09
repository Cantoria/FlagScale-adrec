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

"""Building blocks shared by the recommendation models.

Only standard ``torch.nn`` modules are used, so the same code runs on every
device that Torch-FL exposes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class EagerSelfAttention(nn.Module):
    """Explicit Q/K/V attention without the fused SDPA kernel."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"hidden size {dim} is not divisible by {heads} heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, dim = query.shape
        q = self.q(query).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(key_value).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(key_value).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = self.drop(torch.softmax(scores, dim=-1))
        out = torch.matmul(weights, v).transpose(1, 2).reshape(batch, length, dim)
        return self.o(out)


class TowerMLP(nn.Module):
    """Linear -> BatchNorm1d -> activation (-> Dropout) per hidden layer.

    Matches torch-rechub's ``MLP(output_layer=False)`` used by its DSSM towers.
    """

    def __init__(self, in_dim: int, hidden: Sequence[int], activation: str = "prelu", dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for width in hidden:
            layers.append(nn.Linear(dim, width))
            layers.append(nn.BatchNorm1d(width))
            if activation == "prelu":
                layers.append(nn.PReLU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            else:
                raise ValueError(f"Unsupported activation {activation!r}")
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = width
        self.net = nn.Sequential(*layers)
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ["EagerSelfAttention", "TowerMLP"]
