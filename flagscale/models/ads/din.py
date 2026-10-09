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

"""DIN: Deep Interest Network (Zhou et al., KDD 2018).

Reproduces FuxiCTR v2.0.1 ``model_zoo/DIN`` as configured by the BARS
benchmark ``DIN_amazonelectronics_x1`` (5,403,522 parameters):

* item_id and cate_id tables (padding index 0) are shared by the target and
  the behaviour sequence; the target is ``[item_id; cate_id]``;
* ``DIN_Attention``: MLP over ``[q, k, q - k, q * k]`` with Dice activations,
  masked and softmax-normalized (``din_use_softmax=True``) weights;
* the DNN sees ``[item_id, cate_id, pooled item_history, pooled cate_history]``
  (user_id is only a grouping key for gAUC) with BatchNorm, Dice and dropout.
"""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from .base import AdsModel
from .layers import MLPBlock
from .registry import register_ads_model


class DINAttention(nn.Module):
    """FuxiCTR ``DIN_Attention``."""

    def __init__(self, dim: int, hidden: list[int], activation: str, dropout: float, use_softmax: bool) -> None:
        super().__init__()
        self.dim = dim
        self.use_softmax = use_softmax
        self.attention_layer = MLPBlock(4 * dim, hidden, activation, dropout, batch_norm=False, output_dim=1)

    def forward(self, target: torch.Tensor, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        length = sequence.size(1)
        target = target.unsqueeze(1).expand(-1, length, -1)
        features = torch.cat([target, sequence, target - sequence, target * sequence], dim=-1)
        weight = self.attention_layer(features.view(-1, 4 * self.dim)).view(-1, length)
        mask = mask.to(weight.dtype)
        weight = weight * mask
        if self.use_softmax:
            weight = (weight + -1.0e9 * (1 - mask)).softmax(dim=-1)
        return (weight.unsqueeze(-1) * sequence).sum(dim=1)


@register_ads_model("din")
class DIN(AdsModel):
    def __init__(self, model_cfg: DictConfig, meta: dict[str, Any]) -> None:
        super().__init__(model_cfg, meta)
        dim = int(model_cfg.get("embedding_dim", 64))
        self.dim = dim
        self.embedding_layer = nn.ModuleDict({
            "item_id": nn.Embedding(int(meta["vocab_sizes"]["item_id"]), dim, padding_idx=0),
            "cate_id": nn.Embedding(int(meta["vocab_sizes"]["cate_id"]), dim, padding_idx=0),
        })
        self.attention = DINAttention(
            2 * dim,
            list(model_cfg.get("attention_hidden_units", [512, 256])),
            str(model_cfg.get("attention_hidden_activations", "dice")),
            float(model_cfg.get("attention_dropout", 0.1)),
            bool(model_cfg.get("din_use_softmax", True)),
        )
        self.dnn = MLPBlock(
            4 * dim,
            list(model_cfg.get("dnn_hidden_units", [1024, 512, 256])),
            str(model_cfg.get("dnn_activations", "dice")),
            float(model_cfg.get("net_dropout", 0.1)),
            batch_norm=bool(model_cfg.get("batch_norm", True)),
            output_dim=1,
        )
        self.reset_parameters()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        item, cate = self.embedding_layer["item_id"], self.embedding_layer["cate_id"]
        target = torch.cat([item(batch["item_id"]), cate(batch["cate_id"])], dim=-1)
        sequence = torch.cat([item(batch["item_history"]), cate(batch["cate_history"])], dim=-1)
        pooled = self.attention(target, sequence, batch["item_history"] != 0)
        return self.dnn(torch.cat([target, pooled], dim=-1)).squeeze(-1)
