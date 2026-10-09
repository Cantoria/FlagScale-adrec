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

"""Build the SASRec MovieLens-1M splits.

Protocol of Kang & McAuley (ICDM 2018):

* every rating is an implicit interaction, ordered by timestamp per user;
* items with fewer than 5 interactions are dropped (6,040 users, 3,416 items);
* leave-one-out: the last item is the test target, the second to last the
  validation target, everything before is training;
* evaluation ranks the target against 100 items the user never interacted
  with.  The negatives are drawn once here with a fixed seed so every device
  is evaluated on identical candidate lists.

Output (``--output``): ``train.npz``, ``validation.npz``, ``test.npz`` and
``meta.json``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def left_pad(sequence: list[int], length: int) -> np.ndarray:
    row = np.zeros(length, dtype=np.int32)
    tail = sequence[-length:]
    if tail:
        row[-len(tail):] = tail
    return row


def sample_negatives(rng: np.random.Generator, seen: set[int], n_items: int, count: int) -> list[int]:
    negatives: list[int] = []
    chosen: set[int] = set()
    while len(negatives) < count:
        item = int(rng.integers(1, n_items + 1))
        if item in seen or item in chosen:
            continue
        chosen.add(item)
        negatives.append(item)
    return negatives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", default="/flagos-rec/datasets/raw/movielens/ml-1m/ratings.dat")
    parser.add_argument("--output", default="/flagos-rec/datasets/movielens_1m_sasrec")
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--eval-negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows: list[tuple[int, int, int, int]] = []
    with open(args.ratings, "r", encoding="latin-1") as file:
        for order, line in enumerate(file):
            user, item, _rating, timestamp = line.strip().split("::")
            rows.append((int(user), int(timestamp), order, int(item)))
    item_counts = Counter(item for _, _, _, item in rows)
    rows = [row for row in rows if item_counts[row[3]] >= args.min_count]
    user_counts = Counter(user for user, _, _, _ in rows)
    rows = [row for row in rows if user_counts[row[0]] >= args.min_count]
    # Stable order: timestamp, then file order for interactions in the same second.
    rows.sort()

    item_map = {item: index + 1 for index, item in enumerate(sorted({row[3] for row in rows}))}
    user_map = {user: index for index, user in enumerate(sorted({row[0] for row in rows}))}
    sequences: dict[int, list[int]] = defaultdict(list)
    for user, _, _, item in rows:
        sequences[user_map[user]].append(item_map[item])
    n_users, n_items = len(user_map), len(item_map)

    rng = np.random.default_rng(args.seed)
    length = args.max_len
    train = {k: [] for k in ("user", "seq", "pos")}
    evaluation = {split: {k: [] for k in ("user", "seq", "candidates")} for split in ("validation", "test")}
    user_items_ptr = [0]
    user_items: list[int] = []
    for user in range(n_users):
        sequence = sequences[user]
        seen = set(sequence)
        user_items.extend(sorted(seen))
        user_items_ptr.append(len(user_items))
        history = sequence[:-2]
        train["user"].append(user)
        train["seq"].append(left_pad(history[:-1], length))
        train["pos"].append(left_pad(history[1:], length))
        for split, inputs, target in (
            ("validation", history, sequence[-2]),
            ("test", sequence[:-1], sequence[-1]),
        ):
            evaluation[split]["user"].append(user)
            evaluation[split]["seq"].append(left_pad(inputs, length))
            candidates = [target] + sample_negatives(rng, seen, n_items, args.eval_negatives)
            evaluation[split]["candidates"].append(np.asarray(candidates, dtype=np.int32))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "train.npz",
        user=np.asarray(train["user"], dtype=np.int32),
        seq=np.stack(train["seq"]),
        pos=np.stack(train["pos"]),
        user_items_ptr=np.asarray(user_items_ptr, dtype=np.int64),
        user_items=np.asarray(user_items, dtype=np.int32),
    )
    for split, columns in evaluation.items():
        np.savez(
            output / f"{split}.npz",
            user=np.asarray(columns["user"], dtype=np.int32),
            seq=np.stack(columns["seq"]),
            candidates=np.stack(columns["candidates"]),
        )
    lengths = [len(sequences[user]) for user in range(n_users)]
    meta = {
        "name": "movielens-1m",
        "task": "sequential",
        "n_users": n_users,
        "n_items": n_items,
        "max_len": length,
        "interactions": len(rows),
        "avg_sequence_length": float(np.mean(lengths)),
        "eval_negatives": args.eval_negatives,
        "min_count": args.min_count,
        "seed": args.seed,
        "source": str(Path(args.ratings).resolve()),
    }
    with (output / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
