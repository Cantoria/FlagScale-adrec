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

"""Convert the LightGCN Gowalla split to ``.npz``.

Input is the authors' split (kuandeng/LightGCN ``Data/gowalla/{train,test}.txt``;
the same files are published as reczoo/Gowalla_m1).  Each line is
``user item item ...``.  No filtering or re-splitting is applied; the script
only checks the published statistics (29,858 users, 40,981 items,
810,128 train and 217,242 test interactions).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXPECTED = {"n_users": 29858, "n_items": 40981, "train": 810128, "test": 217242}


def read_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    users, items = [], []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            tokens = line.split()
            if len(tokens) < 2:
                continue
            user = int(tokens[0])
            for token in tokens[1:]:
                users.append(user)
                items.append(int(token))
    return np.asarray(users, dtype=np.int32), np.asarray(items, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="/flagos-rec/datasets/raw/gowalla_official")
    parser.add_argument("--output", default="/flagos-rec/datasets/gowalla_lightgcn")
    args = parser.parse_args()
    raw = Path(args.raw)
    train_users, train_items = read_pairs(raw / "train.txt")
    test_users, test_items = read_pairs(raw / "test.txt")
    n_users = int(max(train_users.max(), test_users.max()) + 1)
    n_items = int(max(train_items.max(), test_items.max()) + 1)
    stats = {"n_users": n_users, "n_items": n_items, "train": len(train_users), "test": len(test_users)}
    mismatch = {k: (stats[k], v) for k, v in EXPECTED.items() if stats[k] != v}
    if mismatch:
        raise ValueError(f"Gowalla statistics differ from the published split: {mismatch}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "train.npz", user=train_users, item=train_items)
    np.savez(output / "test.npz", user=test_users, item=test_items)
    meta = {"name": "gowalla-lightgcn", "task": "graph", **stats, "source": str(raw.resolve())}
    with (output / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
