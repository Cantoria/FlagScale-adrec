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

"""LightGCN (He et al., SIGIR 2020).

Follows the authors' PyTorch implementation (gusye1234/LightGCN-PyTorch,
``code/model.py``): N(0, 0.1) initialization, propagation over the symmetric
normalized user-item adjacency ``D^-1/2 A D^-1/2`` without self loops, and the
mean of the K+1 layer outputs as the final representation.

``propagation`` selects how one propagation step is computed:

* ``sparse`` (default, as in the reference): ``torch.sparse.mm`` with a COO
  adjacency;
* ``index_add``: gather the neighbours with ``index_select`` and aggregate
  with ``index_add``.  Mathematically identical; used on backends without
  sparse matrix kernels.
"""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from .base import RecModel
from .registry import register_rec_model


@register_rec_model("lightgcn")
class LightGCN(RecModel):
    """Graph collaborative filtering; ``forward`` returns BPR scores."""

    task = "graph"

    def __init__(self, model_cfg: DictConfig, meta: dict[str, Any]) -> None:
        super().__init__(model_cfg, meta)
        self.n_users = int(meta["n_users"])
        self.n_items = int(meta["n_items"])
        self.dim = int(model_cfg.get("embedding_dim", 64))
        self.layers = int(model_cfg.get("num_layers", 3))
        self.propagation = str(model_cfg.get("propagation", "sparse"))
        if self.propagation not in {"sparse", "index_add"}:
            raise ValueError(f"Unsupported propagation {self.propagation!r}")
        self.embedding_user = nn.Embedding(self.n_users, self.dim)
        self.embedding_item = nn.Embedding(self.n_items, self.dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        # Edge list of the normalized adjacency, filled by ``set_graph``.
        self.register_buffer("edge_src", torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer("edge_dst", torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer("edge_weight", torch.zeros(0), persistent=False)
        self._adjacency: torch.Tensor | None = None

    def set_graph(self, src: torch.Tensor, dst: torch.Tensor, weight: torch.Tensor) -> None:
        """Install the normalized adjacency as (row, column, value) triples."""

        device = self.embedding_user.weight.device
        self.edge_src = src.to(device)
        self.edge_dst = dst.to(device)
        self.edge_weight = weight.to(device=device, dtype=self.embedding_user.weight.dtype)
        if self.propagation == "sparse":
            n = self.n_users + self.n_items
            indices = torch.stack([self.edge_dst, self.edge_src])
            self._adjacency = torch.sparse_coo_tensor(indices, self.edge_weight, (n, n)).coalesce()

    def _step(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.propagation == "sparse":
            return torch.sparse.mm(self._adjacency, embedding)
        messages = embedding.index_select(0, self.edge_src) * self.edge_weight.unsqueeze(-1)
        return torch.zeros_like(embedding).index_add(0, self.edge_dst, messages)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        outputs = [embedding]
        for _ in range(self.layers):
            embedding = self._step(embedding)
            outputs.append(embedding)
        final = torch.stack(outputs, dim=1).mean(dim=1)
        return final[: self.n_users], final[self.n_users :]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        users, items = self.propagate()
        u = users[batch["user"]]
        pos = items[batch["pos"]]
        neg = items[batch["neg"]]
        return {
            "pos_scores": (u * pos).sum(-1),
            "neg_scores": (u * neg).sum(-1),
            # L2 term on the layer-0 embeddings of the batch, as in bpr_loss().
            "reg": 0.5
            * (
                self.embedding_user(batch["user"]).pow(2).sum()
                + self.embedding_item(batch["pos"]).pow(2).sum()
                + self.embedding_item(batch["neg"]).pow(2).sum()
            )
            / batch["user"].shape[0],
        }

    @torch.no_grad()
    def all_scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.propagate()
