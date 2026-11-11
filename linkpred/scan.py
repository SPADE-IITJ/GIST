"""Per-community error scan.

When and where from one statistic: per-community error rates. The trigger is the largest community z (a scan
statistic); the region is the set of communities whose z is high. Here an "error" is a scored link the model
misses (outside its top 20 against the random pool), counted for both endpoints' communities.
"""
from __future__ import annotations

from typing import List

import numpy as np


class CommunityScan:
    def __init__(self, fast_halflife: float = 8.0, slow_halflife: float = 300.0, min_support: float = 4.0) -> None:
        self._fd = 0.5 ** (1.0 / fast_halflife)
        self._sd = 0.5 ** (1.0 / slow_halflife)
        self.min_support = float(min_support)
        self.fw = self.fn = self.sw = self.sn = np.zeros(0)

    def update(self, wrong: np.ndarray, cids: np.ndarray) -> None:
        wrong, cids = np.asarray(wrong, dtype=float), np.asarray(cids, dtype=np.int64)
        keep = cids >= 0
        wrong, cids = wrong[keep], cids[keep]
        size = max(int(cids.max(initial=-1)) + 1, self.fw.size)
        if size > self.fw.size:
            grow = lambda a: np.concatenate([a, np.zeros(size - a.size)])
            self.fw, self.fn, self.sw, self.sn = map(grow, (self.fw, self.fn, self.sw, self.sn))
        w = np.bincount(cids, weights=wrong, minlength=size)
        n = np.bincount(cids, minlength=size).astype(float)
        self.fw, self.fn = self.fw * self._fd + w, self.fn * self._fd + n
        self.sw, self.sn = self.sw * self._sd + w, self.sn * self._sd + n

    def scores(self) -> np.ndarray:
        """z of each community's recent error rate above its own slow rate; -inf where support is too thin."""
        z = np.full(self.fw.size, -np.inf)
        ok = self.fn >= self.min_support
        if ok.any():
            rate = self.fw[ok] / self.fn[ok]
            ref = (self.sw[ok] + 1.0) / (self.sn[ok] + 2.0)
            z[ok] = (rate - ref) / np.sqrt(ref * (1 - ref) / self.fn[ok])
        return z

    def max_z(self) -> float:
        z = self.scores()
        return float(z.max()) if z.size and np.isfinite(z).any() else 0.0

    def region(self, threshold: float) -> List[int]:
        """Communities above ``threshold``, highest first; the top one at least."""
        z = self.scores()
        if not z.size or not np.isfinite(z).any():
            return []
        order = np.argsort(-z)
        sel = [int(c) for c in order if z[c] > threshold]
        return sel or [int(order[0])]
