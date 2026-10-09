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

"""DeepFM (Guo et al., IJCAI 2017).

Reproduces FuxiCTR v1.1.0 ``DeepFM`` as configured by the BARS benchmark
``DeepFM_criteo_x1``: FM layer (first-order LR embeddings without bias plus
the second-order ``0.5 * ((sum v)^2 - sum v^2)``) added to a DNN over the
flattened field embeddings.
"""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from .base import AdsModel
from .layers import FieldEmbedding, MLPBlock
from .registry import register_ads_model


class LRLayer(nn.Module):
    """First-order term: 1-dim field embeddings summed (no bias in DeepFM)."""

    def __init__(self, num_numeric: int, vocab_sizes: list[int]) -> None:
        super().__init__()
        self.embedding_layer = FieldEmbedding(num_numeric, vocab_sizes, 1)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        return self.embedding_layer(dense, sparse).sum(dim=(1, 2))


@register_ads_model("deepfm")
class DeepFM(AdsModel):
    def __init__(self, model_cfg: DictConfig, meta: dict[str, Any]) -> None:
        super().__init__(model_cfg, meta)
        dim = int(model_cfg.get("embedding_dim", 10))
        vocab = [int(v) for v in meta["vocab_sizes"]]
        num_numeric = int(meta["num_numeric"])
        self.embedding_layer = FieldEmbedding(num_numeric, vocab, dim)
        self.lr_layer = LRLayer(num_numeric, vocab)
        self.dnn = MLPBlock(
            dim * self.embedding_layer.num_fields,
            list(model_cfg.get("hidden_units", [400, 400, 400])),
            str(model_cfg.get("hidden_activations", "relu")),
            float(model_cfg.get("net_dropout", 0.1)),
            batch_norm=bool(model_cfg.get("batch_norm", True)),
            output_dim=1,
        )
        self.reset_parameters()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        dense, sparse = batch["dense"], batch["sparse"]
        emb = self.embedding_layer(dense, sparse)
        second_order = 0.5 * (emb.sum(dim=1).pow(2) - emb.pow(2).sum(dim=1)).sum(dim=-1)
        first_order = self.lr_layer(dense, sparse)
        return first_order + second_order + self.dnn(emb.flatten(start_dim=1)).squeeze(-1)
