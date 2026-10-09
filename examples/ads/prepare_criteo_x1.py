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

"""Encode BARS Criteo_x1 for DeepFM and DCN-Mix.

Input: reczoo/Criteo_x1 ``train.csv`` / ``valid.csv`` / ``test.csv`` (random
7:2:1 split following AFN; md5 30b89c1c... / f73c71fb... / 2c48b26e...).
Columns: ``label, I1..I13`` (already normalized numeric values) and
``C1..C26`` (integer-coded categories).

Encoding follows FuxiCTR v1.1.0's FeatureEncoder for the BARS config
(min_categr_count 1, numeric fields without normalizer):

* numeric values are used as they are; missing values become 0;
* each categorical field has its own vocabulary fitted on train, ordered by
  (-count, token); index 0 is OOV (unseen or missing), known tokens are 1..n,
  vocab_size = n + 1 and there is no padding index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = [f"I{i}" for i in range(1, 14)]
CATEGORICAL = [f"C{i}" for i in range(1, 27)]


def fit_field(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted unique tokens, their 1-based ids) in FuxiCTR order."""

    present = values[~np.isnan(values)]
    tokens, counts = np.unique(present, return_counts=True)
    order = np.lexsort((tokens, -counts))  # by -count, then token
    ids = np.empty(len(tokens), dtype=np.int64)
    ids[order] = np.arange(1, len(tokens) + 1)
    return tokens, ids


def encode_field(values: np.ndarray, tokens: np.ndarray, ids: np.ndarray) -> np.ndarray:
    position = np.searchsorted(tokens, values)
    position = np.clip(position, 0, len(tokens) - 1)
    found = (tokens[position] == values) & ~np.isnan(values)
    return np.where(found, ids[position], 0).astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="/flagos-ads/datasets/raw/Criteo_x1")
    parser.add_argument("--output", default="/flagos-ads/datasets/criteo_x1")
    args = parser.parse_args()
    raw = Path(args.raw)
    dtype = {"label": np.float32, **{c: np.float64 for c in NUMERIC + CATEGORICAL}}
    frames = {split: pd.read_csv(raw / f"{split}.csv", dtype=dtype) for split in ("train", "valid", "test")}
    for split, frame in frames.items():
        print(split, len(frame), "numeric NaN:", int(frame[NUMERIC].isna().sum().sum()),
              "categorical NaN:", int(frame[CATEGORICAL].isna().sum().sum()), flush=True)

    vocabularies = [fit_field(frames["train"][column].to_numpy()) for column in CATEGORICAL]
    vocab_sizes = [len(tokens) + 1 for tokens, _ in vocabularies]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    oov_rate = {}
    for split, frame in frames.items():
        dense = frame[NUMERIC].fillna(0).to_numpy(np.float32)
        sparse = np.stack([encode_field(frame[column].to_numpy(), *vocabularies[i])
                           for i, column in enumerate(CATEGORICAL)], axis=1)
        oov_rate[split] = float((sparse == 0).mean())
        np.savez(output / f"{split}.npz", label=frame["label"].to_numpy(np.int32), dense=dense, sparse=sparse)
        print(split, dense.shape, sparse.shape, "oov rate", oov_rate[split], flush=True)
    meta = {
        "name": "criteo_x1",
        "num_numeric": len(NUMERIC),
        "vocab_sizes": vocab_sizes,
        "total_vocab": int(sum(vocab_sizes)),
        "rows": {split: len(frame) for split, frame in frames.items()},
        "positive_ratio": {split: float(frame["label"].mean()) for split, frame in frames.items()},
        "oov_rate": oov_rate,
        "source": str(raw.resolve()),
    }
    with (output / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "vocab_sizes"}, indent=2))


if __name__ == "__main__":
    main()
