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

"""Build the DSSM MovieLens-1M splits following torch-rechub.

Reference: datawhalechina/torch-rechub ``examples/matching/data/ml-1m/
preprocess_ml.py`` and ``examples/matching/run_ml_dssm.py`` with
``torch_rechub.utils.match.generate_seq_feature_match`` /
``gen_model_input``:

* ratings, movies and users are merged; ``cate_id`` is the first genre;
* user_id, movie_id, gender, age, occupation, zip and cate_id are label-encoded
  to 1..n (vocabulary size n + 1);
* interactions are ordered by timestamp; for each user and each position
  ``i >= 1`` the history is the items before ``i``; the last position is the
  test sample, every other position is a positive training sample followed by
  ``neg_ratio`` = 3 negatives;
* negatives are drawn with probability proportional to ``count ** 0.75``
  (``sample_method=1``), from one pre-generated list consumed in order;
* histories are left-padded and truncated to the most recent 50 items.

Deviations from the reference, both for reproducibility: the time sort is
stable (``mergesort``) and the negative list uses a fixed numpy seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPARSE = ["user_id", "movie_id", "gender", "age", "occupation", "zip", "cate_id"]
USER_COLS = ["user_id", "gender", "age", "occupation", "zip"]


def pad_pre(sequence: list[int], length: int) -> np.ndarray:
    row = np.zeros(length, dtype=np.int32)
    tail = sequence[-length:]
    if tail:
        row[-len(tail):] = tail
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="/flagos-rec/datasets/raw/movielens/ml-1m")
    parser.add_argument("--output", default="/flagos-rec/datasets/movielens_1m_dssm")
    parser.add_argument("--neg-ratio", type=int, default=3)
    parser.add_argument("--seq-max-len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()
    raw = Path(args.raw)

    users = pd.read_csv(raw / "users.dat", sep="::", header=None, engine="python", encoding="ISO-8859-1",
                        names=["user_id", "gender", "age", "occupation", "zip"])
    ratings = pd.read_csv(raw / "ratings.dat", sep="::", header=None, engine="python", encoding="ISO-8859-1",
                          names=["user_id", "movie_id", "rating", "timestamp"])
    movies = pd.read_csv(raw / "movies.dat", sep="::", header=None, engine="python", encoding="ISO-8859-1",
                         names=["movie_id", "title", "genres"])
    data = pd.merge(pd.merge(ratings, movies), users)
    data["cate_id"] = data["genres"].apply(lambda value: value.split("|")[0])
    vocab_sizes = {}
    for column in SPARSE:
        codes, uniques = pd.factorize(data[column], sort=True)  # LabelEncoder order
        data[column] = codes + 1
        vocab_sizes[column] = int(data[column].max() + 1)

    user_profile = data[USER_COLS].drop_duplicates("user_id").set_index("user_id")
    item_profile = data[["movie_id", "cate_id"]].drop_duplicates("movie_id")
    item_cate = dict(zip(item_profile["movie_id"], item_profile["cate_id"]))

    data = data.sort_values("timestamp", kind="mergesort")
    counts = data["movie_id"].value_counts()
    order = sorted(counts.items(), key=lambda kv: -kv[1])
    items = np.array([item for item, _ in order])
    weights = np.array([count for _, count in order], dtype=np.float64) ** 0.75
    rng = np.random.default_rng(args.seed)
    negatives = rng.choice(items, size=len(data) * args.neg_ratio, replace=True, p=weights / weights.sum())

    train_rows: list[tuple[int, int, np.ndarray, int]] = []
    test_rows: list[tuple[int, int, np.ndarray]] = []
    cursor = 0
    for user, group in data.groupby("user_id", sort=True):
        positives = group["movie_id"].tolist()
        for i in range(1, len(positives)):
            history = pad_pre(positives[:i], args.seq_max_len)
            if i != len(positives) - 1:
                train_rows.append((user, positives[i], history, 1))
                for _ in range(args.neg_ratio):
                    train_rows.append((user, int(negatives[cursor]), history, 0))
                    cursor += 1
            else:
                test_rows.append((user, positives[i], history))
    shuffle = rng.permutation(len(train_rows))
    train_rows = [train_rows[index] for index in shuffle]

    def columns(rows, with_label: bool) -> dict[str, np.ndarray]:
        user_ids = np.array([row[0] for row in rows], dtype=np.int32)
        movie_ids = np.array([row[1] for row in rows], dtype=np.int32)
        out = {
            "user_id": user_ids,
            **{col: user_profile.loc[user_ids, col].to_numpy(dtype=np.int32) for col in USER_COLS[1:]},
            "hist_movie_id": np.stack([row[2] for row in rows]),
            "movie_id": movie_ids,
            "cate_id": np.array([item_cate[m] for m in movie_ids], dtype=np.int32),
        }
        if with_label:
            out["label"] = np.array([row[3] for row in rows], dtype=np.int32)
        return out

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "train.npz", **columns(train_rows, True))
    np.savez(output / "test.npz", **columns(test_rows, False))
    np.savez(output / "items.npz", movie_id=item_profile["movie_id"].to_numpy(dtype=np.int32),
             cate_id=item_profile["cate_id"].to_numpy(dtype=np.int32))
    meta = {
        "name": "movielens-1m-torch-rechub-match",
        "task": "two_tower",
        "vocab_sizes": vocab_sizes,
        "n_users": int(data["user_id"].nunique()),
        "n_items": int(len(item_profile)),
        "interactions": int(len(data)),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "neg_ratio": args.neg_ratio,
        "seq_max_len": args.seq_max_len,
        "seed": args.seed,
        "source": str(raw.resolve()),
    }
    with (output / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
