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

"""Registry and factory for native advertising (CTR) models."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig

if TYPE_CHECKING:
    from .base import AdsModel

ADS_MODEL_REGISTRY: dict[str, type["AdsModel"]] = {}


def register_ads_model(name: str) -> Callable[[type["AdsModel"]], type["AdsModel"]]:
    """Register a model under its Hydra ``model.model_type`` name."""

    def decorator(cls: type["AdsModel"]) -> type["AdsModel"]:
        if name in ADS_MODEL_REGISTRY and ADS_MODEL_REGISTRY[name] is not cls:
            raise ValueError(f"Ads model '{name}' is already registered")
        ADS_MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_ads_model(name: str, model_cfg: DictConfig, meta: dict[str, Any], device: torch.device) -> "AdsModel":
    """Instantiate a registered model with the dataset's vocabulary sizes."""

    if name not in ADS_MODEL_REGISTRY:
        raise ValueError(f"Unknown ads model '{name}'. Available: {sorted(ADS_MODEL_REGISTRY)}")
    return ADS_MODEL_REGISTRY[name].from_config(model_cfg, meta, device)
