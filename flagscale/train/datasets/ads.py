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

"""In-memory datasets for the advertising example (examples/ads).

``examples/ads/prepare_*.py`` encode the BARS splits into ``.npz`` arrays
(train / valid / test); they are loaded fully into host memory and every
``__getitem__`` returns a whole batch for a list of row indices.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


def load_meta(path: str | Path) -> dict[str, Any]:
    with (Path(path) / "meta.json").open("r", encoding="utf-8") as file:
        return json.load(file)


class BatchIndexDataset(Dataset[dict[str, torch.Tensor]]):
    """Return a batch of rows for a list of indices."""

    def __init__(self, path: str | Path, split: str, max_rows: int | None = None) -> None:
        arrays = np.load(Path(path) / f"{split}.npz")
        self.tensors = {}
        for key in arrays.files:
            value = arrays[key][: int(max_rows)] if max_rows else arrays[key]
            dtype = np.float32 if value.dtype.kind == "f" else np.int64
            self.tensors[key] = torch.from_numpy(value.astype(dtype, copy=False))
        self.rows = len(self.tensors["label"])

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        index = torch.as_tensor(indices, dtype=torch.long)
        return {key: value[index] for key, value in self.tensors.items()}


def batch_loader(dataset: Dataset, batch_size: int, shuffle: bool, pin_memory: bool, num_workers: int = 0,
                 seed: int = 0, distributed: bool = False) -> DataLoader:
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed, drop_last=False)
    elif shuffle:
        sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(dataset, sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=False),
                      batch_size=None, num_workers=max(0, int(num_workers)), pin_memory=pin_memory,
                      persistent_workers=False)


def build_ads_loaders(data_cfg: DictConfig, batch_size: int, eval_batch_size: int, pin_memory: bool,
                      seed: int) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    """Train / validation / test loaders.

    ``data.valid_split`` names the split used for model selection.  BARS'
    AmazonElectronics_x1 config sets valid_data to test.csv, which
    ``valid_split: test`` reproduces.
    """

    path = str(data_cfg.path)
    meta = load_meta(path)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    workers = int(data_cfg.get("num_workers", 0) or 0)
    train = BatchIndexDataset(path, "train", data_cfg.get("max_samples"))
    eval_max = data_cfg.get("eval_max_samples")
    valid = BatchIndexDataset(path, str(data_cfg.get("valid_split", "valid")), eval_max)
    test = BatchIndexDataset(path, "test", eval_max)
    return (
        batch_loader(train, batch_size, True, pin_memory, workers, seed, distributed),
        batch_loader(valid, eval_batch_size, False, pin_memory),
        batch_loader(test, eval_batch_size, False, pin_memory),
        meta,
    )


__all__ = ["BatchIndexDataset", "batch_loader", "build_ads_loaders", "load_meta"]
