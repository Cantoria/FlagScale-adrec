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

"""Encode BARS AmazonElectronics_x1 for DIN.

Input: reczoo/AmazonElectronics_x1 ``train.csv`` / ``test.csv`` (the DIN
authors' Electronics protocol; md5 e9bf80b9... / 57a20e82...).  Columns:
``label, user_id, item_id, cate_id, item_history, cate_history`` with ``^``
separated histories.

Encoding follows FuxiCTR v2.0.1's FeatureProcessor for the BARS config:

* vocabularies are fitted on train only (min_categr_count 1); item_history
  shares item_id's vocabulary and cate_history shares cate_id's;
* tokens are ordered by (-count, token); index 0 is padding, known tokens are
  1..n and index n + 1 is OOV, so vocab_size = n + 2;
* sequences keep their first ``max_len`` = 100 tokens and are right-padded
  (FuxiCTR default ``padding="post"``);
* user_id is kept as a raw integer grouping key for gAUC.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def fit_vocab(counter: Counter) -> dict[str, int]:
    tokens = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return {token: index for index, (token, _) in enumerate(tokens, start=1)}


def encode_sequences(series: pd.Series, vocab: dict[str, int], oov: int, max_len: int) -> np.ndarray:
    out = np.zeros((len(series), max_len), dtype=np.int32)
    for row, text in enumerate(series.fillna("").astype(str).tolist()):
        if not text:
            continue
        ids = [vocab.get(token, oov) for token in text.split("^")][:max_len]
        out[row, : len(ids)] = ids
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="/flagos-ads/datasets/raw/AmazonElectronics_x1")
    parser.add_argument("--output", default="/flagos-ads/datasets/amazonelectronics_x1")
    parser.add_argument("--max-len", type=int, default=100)
    args = parser.parse_args()
    raw = Path(args.raw)
    dtypes = {"label": np.int32, "user_id": np.int64, "item_id": str, "cate_id": str,
              "item_history": str, "cate_history": str}
    train = pd.read_csv(raw / "train.csv", dtype=dtypes)
    test = pd.read_csv(raw / "test.csv", dtype=dtypes)

    vocabs = {}
    for target, sequence in (("item_id", "item_history"), ("cate_id", "cate_history")):
        counter = Counter(train[target].tolist())
        for text in train[sequence].fillna("").tolist():
            if text:
                counter.update(text.split("^"))
        vocabs[target] = fit_vocab(counter)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    vocab_sizes = {name: len(vocab) + 2 for name, vocab in vocabs.items()}
    for split, frame in (("train", train), ("test", test)):
        arrays = {"label": frame["label"].to_numpy(np.int32), "user_id": frame["user_id"].to_numpy(np.int64)}
        for target, sequence in (("item_id", "item_history"), ("cate_id", "cate_history")):
            vocab, oov = vocabs[target], vocab_sizes[target] - 1
            arrays[target] = np.array([vocab.get(t, oov) for t in frame[target].tolist()], dtype=np.int32)
            arrays[sequence] = encode_sequences(frame[sequence], vocab, oov, args.max_len)
        np.savez(output / f"{split}.npz", **arrays)
        print(split, {k: v.shape for k, v in arrays.items()}, flush=True)
    meta = {
        "name": "amazonelectronics_x1",
        "vocab_sizes": vocab_sizes,
        "rows": {"train": len(train), "test": len(test)},
        "positive_ratio": {"train": float(train["label"].mean()), "test": float(test["label"].mean())},
        "max_len": args.max_len,
        "source": str(raw.resolve()),
    }
    with (output / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
