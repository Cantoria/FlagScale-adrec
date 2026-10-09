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

"""DCN-Mix: DCN-V2 with a mixture of low-rank experts (Wang et al., WWW 2021).

Reproduces FuxiCTR v1.1.0 ``DCNv2`` with ``use_low_rank_mixture=True`` and the
BARS ``DCNv2_criteo_x1`` settings otherwise: parallel structure, CrossNetMix
(3 layers, rank 32, 4 experts) next to a 3x500 ReLU DNN, concatenated into one
logit.  BARS publishes only the full-rank variant on Criteo_x1; this model
switches on the mixture exactly as FuxiCTR's flag does.
"""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from .base import AdsModel
from .layers import CrossNetMix, FieldEmbedding, MLPBlock
from .registry import register_ads_model


@register_ads_model("dcn_mix")
class DCNMix(AdsModel):
    def __init__(self, model_cfg: DictConfig, meta: dict[str, Any]) -> None:
        super().__init__(model_cfg, meta)
        dim = int(model_cfg.get("embedding_dim", 10))
        vocab = [int(v) for v in meta["vocab_sizes"]]
        self.embedding_layer = FieldEmbedding(int(meta["num_numeric"]), vocab, dim)
        input_dim = dim * self.embedding_layer.num_fields
        self.crossnet = CrossNetMix(
            input_dim,
            int(model_cfg.get("num_cross_layers", 3)),
            int(model_cfg.get("low_rank", 32)),
            int(model_cfg.get("num_experts", 4)),
        )
        self.parallel_dnn = MLPBlock(
            input_dim,
            list(model_cfg.get("parallel_dnn_hidden_units", [500, 500, 500])),
            str(model_cfg.get("dnn_activations", "relu")),
            float(model_cfg.get("net_dropout", 0.1)),
            batch_norm=bool(model_cfg.get("batch_norm", False)),
        )
        self.fc = nn.Linear(input_dim + self.parallel_dnn.out_dim, 1)
        self.reset_parameters()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        flat = self.embedding_layer(batch["dense"], batch["sparse"]).flatten(start_dim=1)
        return self.fc(torch.cat([self.crossnet(flat), self.parallel_dnn(flat)], dim=-1)).squeeze(-1)
