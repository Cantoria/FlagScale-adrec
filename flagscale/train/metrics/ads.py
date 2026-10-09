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

"""CTR metrics with FuxiCTR's definitions (fuxictr/metrics.py)."""

from __future__ import annotations

import numpy as np


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for tied scores (same as roc_auc_score)."""

    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [scores.size]])
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def log_loss(labels: np.ndarray, probabilities: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(labels, dtype=np.float64)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def group_auc(groups: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> float:
    """gAUC (DIN paper / FuxiCTR): per-group AUC weighted by group size,
    skipping groups whose labels are all 0 or all 1."""

    order = np.argsort(groups, kind="mergesort")
    groups, labels, scores = np.asarray(groups)[order], np.asarray(labels)[order], np.asarray(scores)[order]
    boundaries = np.flatnonzero(np.diff(groups)) + 1
    total, weight = 0.0, 0
    for y, s in zip(np.split(labels, boundaries), np.split(scores, boundaries)):
        value = roc_auc(y, s)
        if np.isnan(value):
            continue
        total += value * y.size
        weight += y.size
    return float(total / weight) if weight else float("nan")


def ctr_metrics(labels: np.ndarray, probabilities: np.ndarray, groups: np.ndarray | None = None) -> dict[str, float]:
    result = {"AUC": roc_auc(labels, probabilities), "logloss": log_loss(labels, probabilities)}
    if groups is not None:
        result["gAUC"] = group_auc(groups, labels, probabilities)
    return result


__all__ = ["ctr_metrics", "group_auc", "log_loss", "roc_auc"]
