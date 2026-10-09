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

"""Registry and factory for native recommendation models."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig

if TYPE_CHECKING:
    from .base import RecModel

REC_MODEL_REGISTRY: dict[str, type["RecModel"]] = {}


def register_rec_model(name: str) -> Callable[[type["RecModel"]], type["RecModel"]]:
    """Register a model under its Hydra ``model.model_type`` name."""

    def decorator(cls: type["RecModel"]) -> type["RecModel"]:
        if name in REC_MODEL_REGISTRY and REC_MODEL_REGISTRY[name] is not cls:
            raise ValueError(f"Rec model '{name}' is already registered")
        REC_MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_rec_model(name: str, model_cfg: DictConfig, meta: dict[str, Any], device: torch.device) -> "RecModel":
    """Instantiate a registered model with the dataset's vocabulary sizes."""

    if name not in REC_MODEL_REGISTRY:
        raise ValueError(f"Unknown rec model '{name}'. Available: {sorted(REC_MODEL_REGISTRY)}")
    return REC_MODEL_REGISTRY[name].from_config(model_cfg, meta, device)
