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

"""Common contract for the native recommendation models."""

from __future__ import annotations

import json
from abc import ABC
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when native DDP is active."""

    return model.module if isinstance(model, DDP) else model


class RecModel(nn.Module, ABC):
    """Base class for recommendation models trained from scratch on ID features.

    Vocabulary sizes come from the preprocessed dataset's ``meta.json``.  Each
    model declares its ``task``; the training entrypoint dispatches the
    objective, sampling and evaluation protocol on that task.
    """

    task: str = ""

    def __init__(self, model_cfg: DictConfig, meta: dict[str, Any]) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.meta = meta

    @classmethod
    def from_config(cls, model_cfg: DictConfig, meta: dict[str, Any], device: torch.device) -> "RecModel":
        return cls(model_cfg, meta).to(device)

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in self.state_dict().items()}, output_dir / "model.pt")
        with (output_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(
                {"model": OmegaConf.to_container(self.model_cfg, resolve=True), "meta": self.meta},
                file,
                indent=2,
            )
