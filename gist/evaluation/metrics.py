"""Classification metrics, stage timing, and multi-seed aggregation.

(The DASS-GNN prototype shipped this module as an empty file.)
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import Tensor


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


def classification_metrics(preds: Tensor, targets: Tensor) -> ClassificationMetrics | None:
    """Metrics over the supplied nodes, or None when the set is empty.

    Callers must pass *holdout* nodes only. Returning None rather than zeros
    matters: the prototype scored an empty selection as 0.0 accuracy, which
    silently dragged the reported mean down.
    """
    if preds.numel() == 0:
        return None

    y_true = targets.detach().cpu().numpy()
    y_pred = preds.detach().cpu().numpy()
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        support=int(y_true.shape[0]),
    )


class StageTimer:
    """Wall-clock accumulator per pipeline stage.

    Preliminary in Phase 1 - it exists to expose where step time actually goes
    before Phase 3 builds the full latency/throughput/memory instrumentation.
    """

    def __init__(self) -> None:
        self.totals: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] += time.perf_counter() - start
            self.counts[name] += 1

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "total_s": total,
                "calls": self.counts[name],
                "mean_ms": 1000.0 * total / max(self.counts[name], 1),
            }
            for name, total in sorted(self.totals.items())
        }


def recovery_steps(
    accuracies: Sequence[float | None],
    event_steps: Sequence[int],
    *,
    pre_window: int = 30,
    post_window: int = 10,
    tolerance: float = 0.01,
    horizon: int = 400,
) -> List[int | None]:
    """Steps for accuracy to return to its pre-drift level after each event.

    This is the proposal's "recovery adaptivity" metric, which only becomes
    measurable once drift arrives as discrete, irregular events. ``None``
    means the arm had not recovered within ``horizon`` steps.
    """
    series = np.array(
        [np.nan if a is None else a for a in accuracies], dtype=float
    )
    out: List[int | None] = []

    for step in event_steps:
        pre_slice = series[max(0, step - pre_window) : step]
        pre = np.nanmean(pre_slice) if pre_slice.size else np.nan
        if np.isnan(pre):
            out.append(None)
            continue

        target = pre - tolerance
        recovered: int | None = None
        limit = min(len(series), step + horizon)
        for offset in range(step, limit - post_window):
            window = series[offset : offset + post_window]
            if window.size and np.nanmean(window) >= target:
                recovered = offset - step
                break
        out.append(recovered)

    return out


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        # Sample std (ddof=1) - we are reporting over a sample of seeds.
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def aggregate_over_seeds(
    per_seed: Iterable[Dict[str, float]],
    keys: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Collapse a list of per-seed summaries into mean/std per key."""
    rows: List[Dict[str, float]] = list(per_seed)
    return {key: mean_std([row.get(key) for row in rows]) for key in keys}
