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

"""Ranking metrics for the recommendation example."""

from __future__ import annotations

import numpy as np


def sampled_topk(scores: np.ndarray, ks: tuple[int, ...] = (5, 10)) -> dict[str, float]:
    """HR@k / NDCG@k when the ground-truth item is candidate column 0.

    Used for SASRec's 1 positive + 100 sampled negatives protocol.  The rank
    counts candidates scored strictly higher than the target.
    """

    scores = np.asarray(scores, dtype=np.float64)
    rank = (scores[:, 1:] > scores[:, :1]).sum(axis=1)
    metrics: dict[str, float] = {}
    for k in ks:
        hit = rank < k
        metrics[f"hr@{k}"] = float(hit.mean())
        metrics[f"ndcg@{k}"] = float(np.where(hit, 1.0 / np.log2(rank + 2.0), 0.0).mean())
    return metrics


class FullRankAccumulator:
    """Recall / NDCG / precision / hit rate over the full item catalogue.

    ``update`` receives the top-``max_k`` item ids per user and the user's
    ground-truth items; NDCG uses the ideal DCG of ``min(|truth|, k)`` hits,
    which is the definition in LightGCN's ``NDCGatK_r`` and torch-rechub's
    ``topk_metrics``.
    """

    def __init__(self, ks: tuple[int, ...]) -> None:
        self.ks = tuple(sorted(ks))
        self.sums = {f"{name}@{k}": 0.0 for k in self.ks for name in ("recall", "ndcg", "precision", "hit")}
        self.users = 0
        self.discount = 1.0 / np.log2(np.arange(2, max(self.ks) + 2))

    def update(self, topk: np.ndarray, truths: list[np.ndarray]) -> None:
        for predicted, truth in zip(topk, truths):
            if len(truth) == 0:
                continue
            relevant = np.isin(predicted, truth)
            self.users += 1
            for k in self.ks:
                hits = relevant[:k]
                n_hits = float(hits.sum())
                ideal = self.discount[: min(len(truth), k)].sum()
                self.sums[f"recall@{k}"] += n_hits / len(truth)
                self.sums[f"precision@{k}"] += n_hits / k
                self.sums[f"hit@{k}"] += float(n_hits > 0)
                self.sums[f"ndcg@{k}"] += float((hits * self.discount[:k]).sum() / ideal)

    def result(self) -> dict[str, float]:
        return {name: value / max(self.users, 1) for name, value in self.sums.items()}


__all__ = ["FullRankAccumulator", "sampled_topk"]
