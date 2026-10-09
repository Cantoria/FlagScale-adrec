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

"""Layers of the advertising models, following FuxiCTR.

Only standard ``torch.nn`` modules are used, so the same code runs on every
device that Torch-FL exposes.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class Dice(nn.Module):
    """Dice activation (FuxiCTR v2 ``layers.activations.Dice``)."""

    def __init__(self, dim: int, eps: float = 1e-9) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, affine=False, eps=eps, momentum=0.01)
        self.alpha = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(self.bn(x))
        return p * x + self.alpha * (1 - p) * x


def _activation(name: str, width: int) -> nn.Module:
    name = name.lower()
    if name == "dice":
        return Dice(width)
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation {name!r}")


class MLPBlock(nn.Module):
    """Linear -> [BatchNorm1d] -> activation -> [Dropout] per hidden layer.

    Same layer order as FuxiCTR's ``MLP_Block`` (v2) / ``MLP_Layer`` (v1) with
    ``norm_before_activation=True``; ``output_dim`` appends a final Linear.
    """

    def __init__(self, in_dim: int, hidden: Sequence[int], activation: str = "relu", dropout: float = 0.0,
                 batch_norm: bool = False, output_dim: int | None = None) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for width in hidden:
            layers.append(nn.Linear(dim, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(_activation(activation, width))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = width
        if output_dim is not None:
            layers.append(nn.Linear(dim, output_dim))
            dim = output_dim
        self.mlp = nn.Sequential(*layers)
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FieldEmbedding(nn.Module):
    """FuxiCTR v1 ``EmbeddingLayer`` for Criteo-style inputs.

    Each numeric field is ``Linear(1, dim, bias=False)`` applied to its value;
    each categorical field has its own table.  Output: [B, fields, dim] with
    numeric fields first, matching the feature order of the dataset config.
    """

    def __init__(self, num_numeric: int, vocab_sizes: Sequence[int], dim: int) -> None:
        super().__init__()
        self.numeric = nn.ModuleList([nn.Linear(1, dim, bias=False) for _ in range(num_numeric)])
        self.categorical = nn.ModuleList([nn.Embedding(int(v), dim) for v in vocab_sizes])
        self.num_fields = num_numeric + len(vocab_sizes)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        fields = [layer(dense[:, i : i + 1]) for i, layer in enumerate(self.numeric)]
        fields += [table(sparse[:, i]) for i, table in enumerate(self.categorical)]
        return torch.stack(fields, dim=1)


class CrossNetMix(nn.Module):
    """DCN-V2 cross network with a mixture of low-rank experts.

    Same computation and initialization as FuxiCTR v1.1.0 ``CrossNetMix``
    (from DeepCTR-Torch), which TorchRec implements as
    ``LowRankMixtureCrossNet``.
    """

    def __init__(self, in_features: int, layer_num: int, low_rank: int, num_experts: int) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts
        self.U_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, in_features, low_rank))) for _ in range(layer_num)])
        self.V_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, in_features, low_rank))) for _ in range(layer_num)])
        self.C_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, low_rank, low_rank))) for _ in range(layer_num)])
        self.gating = nn.ModuleList([nn.Linear(in_features, 1, bias=False) for _ in range(num_experts)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(in_features, 1)) for _ in range(layer_num)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            experts, gates = [], []
            for expert in range(self.num_experts):
                gates.append(self.gating[expert](x_l.squeeze(2)))
                v_x = torch.tanh(torch.matmul(self.V_list[i][expert].t(), x_l))
                v_x = torch.tanh(torch.matmul(self.C_list[i][expert], v_x))
                uv_x = torch.matmul(self.U_list[i][expert], v_x)
                experts.append((x_0 * (uv_x + self.bias[i])).squeeze(2))
            experts_t = torch.stack(experts, 2)
            gates_t = torch.stack(gates, 1)
            x_l = torch.matmul(experts_t, gates_t.softmax(1)) + x_l
        return x_l.squeeze(2)


__all__ = ["CrossNetMix", "Dice", "FieldEmbedding", "MLPBlock"]
