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

"""DSSM two-tower retrieval model for recommendation.

The original DSSM (Huang et al., CIKM 2013) is a web-search model trained on
proprietary click logs.  The recommendation variant here follows
torch-rechub's ``DSSM`` and its MovieLens-1M example
(``examples/matching/run_ml_dssm.py``):

* user tower input: embeddings of user_id, gender, age, occupation, zip and the
  mean-pooled watch history (sharing the movie_id table), 16 dims each;
* item tower input: embeddings of movie_id and its first genre;
* each tower is Linear-BatchNorm-PReLU with widths [256, 128, 64], and its
  output is L2-normalized;
* the training score is ``sigmoid(<u, v>)`` with point-wise binary
  cross-entropy.  torch-rechub accepts a ``temperature`` argument but its
  forward does not apply it, so ``temperature`` defaults to 1.0 here.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from .base import RecModel
from .layers import TowerMLP
from .registry import register_rec_model

USER_FIELDS = ("user_id", "gender", "age", "occupation", "zip")
ITEM_FIELDS = ("movie_id", "cate_id")


@register_rec_model("dssm")
class DSSM(RecModel):
    """Two-tower model; ``forward`` returns logits of (user, item) pairs."""

    task = "two_tower"

    def __init__(self, model_cfg: DictConfig, meta: dict[str, Any]) -> None:
        super().__init__(model_cfg, meta)
        dim = int(model_cfg.get("embedding_dim", 16))
        vocab = meta["vocab_sizes"]
        self.tables = nn.ModuleDict({name: nn.Embedding(int(vocab[name]), dim) for name in USER_FIELDS + ITEM_FIELDS})
        hidden = list(model_cfg.get("tower_hidden", [256, 128, 64]))
        activation = str(model_cfg.get("activation", "prelu"))
        dropout = float(model_cfg.get("dropout", 0.0))
        self.user_tower = TowerMLP(dim * (len(USER_FIELDS) + 1), hidden, activation, dropout)
        self.item_tower = TowerMLP(dim * len(ITEM_FIELDS), hidden, activation, dropout)
        self.temperature = float(model_cfg.get("temperature", 1.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # torch-rechub initializes every embedding with N(0, 1e-4); linear and
        # BatchNorm layers keep the PyTorch defaults.
        for table in self.tables.values():
            nn.init.normal_(table.weight, 0.0, 1e-4)

    def user_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        fields = [self.tables[name](batch[name]) for name in USER_FIELDS]
        history = batch["hist_movie_id"]
        mask = (history != 0).to(fields[0].dtype)
        pooled = (self.tables["movie_id"](history) * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-16)
        return F.normalize(self.user_tower(torch.cat(fields + [pooled], dim=-1)), p=2, dim=1)

    def item_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        fields = [self.tables[name](batch[name]) for name in ITEM_FIELDS]
        return F.normalize(self.item_tower(torch.cat(fields, dim=-1)), p=2, dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return (self.user_embedding(batch) * self.item_embedding(batch)).sum(-1) / self.temperature
