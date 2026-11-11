"""Natural drift: a stream in which nodes arrive in publication-year order (G2b).

Nothing is injected. Nodes of the base window form the initial graph; every later node arrives at a step fixed by
its year (ties broken by a seeded permutation), and an edge becomes visible once both endpoints have arrived. Batch
roots are drawn only from arrived nodes, half of them from recent arrivals, so the model is scored on new papers.
The rules are pre-registered in docs/g2b-preregistration.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from ..seeding import derive, numpy_rng
from .schedule import StreamPlan


def year_windows(years: np.ndarray, base_fraction: float = 0.3, calibration_years: int = 2) -> Dict[str, Tuple[int, int]]:
    """Base ends at the first year whose cumulative node share reaches ``base_fraction``; calibration is the next
    ``calibration_years`` calendar years; evaluation is everything after. Ranges are inclusive and may be empty."""
    ys, counts = np.unique(years, return_counts=True)
    cum = np.cumsum(counts) / counts.sum()
    base_end = int(ys[np.argmax(cum >= base_fraction)])
    calib_end = base_end + calibration_years
    return {"base": (int(ys.min()), base_end), "calibration": (base_end + 1, calib_end),
            "evaluation": (calib_end + 1, int(ys.max()))}


@dataclass
class TemporalStreamPlan(StreamPlan):
    """A StreamPlan whose steps admit nodes, and the edges they complete, in year order."""

    edge_offsets: Tensor = None  # [T+1]: stream_edges[:, off[t]:off[t+1]] become visible at step t
    arrival_step: Tensor = None  # [N]: the step each node arrives at; -1 for the base window
    node_year: Tensor = None  # [N]
    step_year: Tensor = None  # [T]: year of the latest node arrived by each step
    windows: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return int(self.root_nodes.shape[0])

    def edges_at(self, step: int) -> Tensor:
        return self.stream_edges[:, int(self.edge_offsets[step]) : int(self.edge_offsets[step + 1])]

    def arrived(self, step: int) -> Tensor:
        """Boolean mask of nodes present at ``step`` (the base window is present before step 0)."""
        return self.arrival_step <= step

    def window_of(self, year: int) -> str:
        for name, (lo, hi) in self.windows.items():
            if lo <= year <= hi:
                return name
        return "none"

    def step_windows(self) -> list:
        return [self.window_of(int(y)) for y in self.step_year]


def build_temporal_plan(
    edge_index: Tensor,
    node_year: Tensor,
    num_nodes: int,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    base_fraction: float = 0.3,
    recent_steps: int = 50,
    recent_share: float = 0.5,
) -> TemporalStreamPlan:
    years = node_year.view(-1).cpu().numpy().astype(np.int64)
    windows = year_windows(years, base_fraction)
    base_mask = years <= windows["base"][1]

    # Arrival order: by year, ties broken by a seeded permutation. Rank r arrives at step floor(r * steps / n).
    stream_nodes = np.nonzero(~base_mask)[0]
    tie = numpy_rng(derive(seed, 0x7E)).permutation(stream_nodes.size)
    order = stream_nodes[np.lexsort((tie, years[stream_nodes]))]
    arrival = np.full(num_nodes, -1, dtype=np.int64)
    if order.size:
        arrival[order] = (np.arange(order.size) * steps) // order.size

    # Undirected pairs, as build_plan; each becomes visible when its later endpoint arrives.
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    keep = lo != hi
    pairs = np.unique(np.stack([lo[keep], hi[keep]], axis=1), axis=0)
    visible = np.maximum(arrival[pairs[:, 0]], arrival[pairs[:, 1]])
    base_pairs = pairs[visible < 0]
    stream_pairs, at = pairs[visible >= 0], visible[visible >= 0]
    by_step = np.argsort(at, kind="stable")
    stream_pairs, at = stream_pairs[by_step], at[by_step]
    offsets = np.searchsorted(at, np.arange(steps + 1), side="left")

    # Roots: half from the last `recent_steps` steps' arrivals, the rest uniformly from all arrived nodes.
    root_rng = numpy_rng(derive(seed, 0x40))
    base_nodes = np.nonzero(base_mask)[0]
    pool = np.concatenate([base_nodes, order])  # arrived at step t = pool[: base + arrived_by[t]]
    arr_sorted = arrival[order]
    arrived_by = np.searchsorted(arr_sorted, np.arange(steps), side="right")
    n_roots = int(min(batch_size, base_nodes.size + (arrived_by[0] if steps else 0)))
    n_recent_max = int(round(recent_share * n_roots))
    roots = np.empty((steps, n_roots), dtype=np.int64)
    step_year = np.empty(steps, dtype=np.int64)
    for t in range(steps):
        m = base_nodes.size + int(arrived_by[t])
        first_recent = int(np.searchsorted(arr_sorted, t - recent_steps + 1, side="left"))
        recent = order[first_recent : arrived_by[t]]
        n_rec = min(recent.size, n_recent_max)
        picked = root_rng.choice(recent, size=n_rec, replace=False) if n_rec else np.empty(0, dtype=np.int64)
        cand = pool[root_rng.choice(m, size=min(m, n_roots + n_rec), replace=False)]
        rest = cand[~np.isin(cand, picked)][: n_roots - n_rec]
        roots[t] = np.concatenate([picked, rest])
        step_year[t] = years[order[arrived_by[t] - 1]] if arrived_by[t] else windows["base"][1]

    return TemporalStreamPlan(
        num_nodes=int(num_nodes),
        base_edges=torch.from_numpy(base_pairs.T.copy()).long(),
        stream_edges=torch.from_numpy(stream_pairs.T.copy()).long(),
        drift_events=[],
        events_by_step={},
        root_nodes=torch.from_numpy(roots).long(),
        sample_seeds=[derive(seed, 0x5A, step) for step in range(steps)],
        seed=int(seed),
        edge_offsets=torch.from_numpy(offsets).long(),
        arrival_step=torch.from_numpy(arrival).long(),
        node_year=torch.from_numpy(years).long(),
        step_year=torch.from_numpy(step_year).long(),
        windows=windows,
    )
