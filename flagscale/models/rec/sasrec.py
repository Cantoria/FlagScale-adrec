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

"""SASRec: Self-Attentive Sequential Recommendation (Kang & McAuley, ICDM 2018).

The block layout follows the authors' PyTorch port (SASRec.pytorch): pre-norm
query, residual from the normalized query, point-wise FFN built from
``Conv1d(kernel_size=1)`` and a timeline mask that zeroes padded positions.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from .base import RecModel
from .layers import EagerSelfAttention
from .registry import register_rec_model


class PointWiseFeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop2(self.conv2(self.drop1(torch.relu(self.conv1(x.transpose(1, 2))))))
        return y.transpose(1, 2) + x


@register_rec_model("sasrec")
class SASRec(RecModel):
    """Next-item model; ``forward`` returns positive/negative logits."""

    task = "sequential"

    def __init__(self, model_cfg: DictConfig, meta: dict[str, Any]) -> None:
        super().__init__(model_cfg, meta)
        self.dim = int(model_cfg.get("hidden_size", 50))
        blocks = int(model_cfg.get("num_blocks", 2))
        heads = int(model_cfg.get("num_heads", 1))
        dropout = float(model_cfg.get("dropout", 0.2))
        self.max_len = int(meta["max_len"])
        self.item_emb = nn.Embedding(int(meta["n_items"]) + 1, self.dim, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_len, self.dim)
        self.emb_drop = nn.Dropout(dropout)
        self.attn_norms = nn.ModuleList([nn.LayerNorm(self.dim, eps=1e-8) for _ in range(blocks)])
        self.attns = nn.ModuleList([EagerSelfAttention(self.dim, heads, dropout) for _ in range(blocks)])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(self.dim, eps=1e-8) for _ in range(blocks)])
        self.ffns = nn.ModuleList([PointWiseFeedForward(self.dim, dropout) for _ in range(blocks)])
        self.last_norm = nn.LayerNorm(self.dim, eps=1e-8)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The reference implementation initializes every weight matrix with
        # Xavier-normal; LayerNorm and bias vectors keep their defaults.
        for name, parameter in self.named_parameters():
            if parameter.dim() > 1:
                nn.init.xavier_normal_(parameter)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        length = seq.size(1)
        timeline = (seq != 0).unsqueeze(-1).to(self.item_emb.weight.dtype)
        positions = torch.arange(length, device=seq.device)
        x = self.item_emb(seq) * math.sqrt(self.dim) + self.pos_emb(positions).unsqueeze(0)
        x = self.emb_drop(x) * timeline
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=seq.device), diagonal=1)
        for attn_norm, attn, ffn_norm, ffn in zip(self.attn_norms, self.attns, self.ffn_norms, self.ffns):
            q = attn_norm(x)
            x = q + attn(q, x, causal)
            x = ffn(ffn_norm(x)) * timeline
        return self.last_norm(x)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(batch["seq"])
        pos_logits = (hidden * self.item_emb(batch["pos"])).sum(-1)
        neg_logits = (hidden * self.item_emb(batch["neg"])).sum(-1)
        return pos_logits, neg_logits

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Score the candidate list against the representation of the last
        # position: [B, D] x [B, C, D] -> [B, C].
        last = self.encode(batch["seq"])[:, -1, :]
        return torch.bmm(self.item_emb(batch["candidates"]), last.unsqueeze(-1)).squeeze(-1)
