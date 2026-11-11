"""Supervised drift signal: prediction error on the labelled training roots the stream already reveals.

Why it exists (preliminary campaign): the composite detector compares batch-level summaries (mean
embedding norm, class histogram) against a reference that adapts within ~10 steps. Each batch is a fresh uniform
sample and an injected event touches 3-8% of the nodes, so on the calibration seeds its score had the same
distribution on a drift-free twin stream as on the drifting one (AUC 0.49-0.50 for "just after an event" vs "far
from any"), and its trigger fired on noise, at a near-constant rate. A label remap is also invisible to it by
construction. The error rate on the training roots is not: AUC 0.76-0.88 on the same seeds.

Two uses, sharing one update per step:

- **when** (``drift.signal: error``): a two-window test on the training-root error rate, the last ``error_window``
  steps against the ``error_reference`` steps before them, as a z statistic; it flags above
  ``drift.error_threshold`` (calibrated on a drift-free twin of the stream, seeds 100-102).
- **where** (``drift.region_signal: error``): per community, a recent (half-life ``error_halflife``) and a slow
  (half-life ``error_ref_halflife``) decayed error rate; a community scores the z of its recent rate above its own
  slow rate, and the selected set is the tail (mean + k std), as for the embedding score.

Only training roots (``train_mask``) are used: holdout labels never reach the detector.
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np


class ErrorSignal:
    def __init__(self, window: int = 30, reference: int = 150, threshold: float = 3.0, halflife: float = 50.0,
                 ref_halflife: float = 300.0, selection_k: float = 1.5, min_support: float = 3.0) -> None:
        self.window, self.reference, self.threshold = int(window), int(reference), float(threshold)
        self.selection_k, self.min_support = float(selection_k), float(min_support)
        self._steps: deque = deque(maxlen=self.window + self.reference)
        self._decay = 0.5 ** (1.0 / halflife)
        self._ref_decay = 0.5 ** (1.0 / ref_halflife)
        self._rw = self._rn = self._bw = self._bn = np.zeros(0)
        self.z = 0.0
        self.flag = False

    # -- per step -----------------------------------------------------------------------------------------------
    def update(self, wrong: np.ndarray, communities: Optional[np.ndarray] = None) -> float:
        """``wrong``: bool per training root of this step's batch; ``communities``: their community ids (or None)."""
        wrong = np.asarray(wrong, dtype=bool)
        self._steps.append((int(wrong.sum()), int(wrong.size)))
        self.z, self.flag = self._window_test(), False
        self.flag = bool(len(self._steps) == self._steps.maxlen and self.z > self.threshold)
        if communities is not None:
            self._update_communities(wrong, np.asarray(communities))
        return self.z

    def _window_test(self) -> float:
        if len(self._steps) < self._steps.maxlen:
            return 0.0
        s = np.asarray(self._steps, dtype=float)
        ref, rec = s[: self.reference], s[self.reference:]
        nr, nw = ref[:, 1].sum(), rec[:, 1].sum()
        if nr < 1 or nw < 1:
            return 0.0
        a, b = rec[:, 0].sum() / nw, ref[:, 0].sum() / nr
        p = (rec[:, 0].sum() + ref[:, 0].sum()) / (nw + nr)
        se = np.sqrt(max(p * (1 - p), 1e-6) * (1 / nw + 1 / nr))
        return float((a - b) / se)

    def _update_communities(self, wrong: np.ndarray, cids: np.ndarray) -> None:
        keep = cids >= 0
        wrong, cids = wrong[keep], cids[keep]
        size = int(cids.max(initial=-1)) + 1
        if size > self._rw.size:
            grow = lambda a: np.concatenate([a, np.zeros(size - a.size)])
            self._rw, self._rn, self._bw, self._bn = map(grow, (self._rw, self._rn, self._bw, self._bn))
        w = np.bincount(cids, weights=wrong.astype(float), minlength=self._rw.size)
        n = np.bincount(cids, minlength=self._rw.size).astype(float)
        self._rw = self._rw * self._decay + w
        self._rn = self._rn * self._decay + n
        self._bw = self._bw * self._ref_decay + w
        self._bn = self._bn * self._ref_decay + n

    # -- where --------------------------------------------------------------------------------------------------
    def community_scores(self) -> dict:
        ok = np.nonzero(self._rn >= self.min_support)[0]
        if ok.size == 0:
            return {}
        rate = self._rw[ok] / self._rn[ok]
        ref = (self._bw[ok] + 1.0) / (self._bn[ok] + 2.0)          # Laplace-smoothed slow rate
        z = (rate - ref) / np.sqrt(ref * (1 - ref) / self._rn[ok])
        return {int(c): float(v) for c, v in zip(ok, z)}

    def drifting(self) -> List[int]:
        scores = self.community_scores()
        if len(scores) < 2:
            return list(scores)
        v = np.fromiter(scores.values(), dtype=float)
        cut = float(v.mean() + self.selection_k * v.std())
        sel = [c for c, s in scores.items() if s > cut]
        return sel or [max(scores, key=scores.get)]


class ErrorRegionBackend:
    """A community backend whose ``drifting`` ranks communities by the error signal; everything else delegates."""

    def __init__(self, inner, signal: ErrorSignal) -> None:
        self.inner, self.signal = inner, signal
        self.name = f"{inner.name}+error"

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def drifting(self, observation) -> List[int]:
        return self.signal.drifting()
