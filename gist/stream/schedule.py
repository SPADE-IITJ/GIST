"""The shared stream plan.

This is the central Phase-1 correction. In the DASS-GNN prototype the three
arms ran sequentially off one advancing global RNG stream, and drew their
drift-injection edges from ``random.randint`` and their batches from a
``shuffle=True`` loader. Each arm therefore saw a *different* graph evolution
and a *different* batch sequence, so the reported accuracy gaps conflated the
adaptation policy with sampling noise.

A ``StreamPlan`` is materialised once per seed and replayed verbatim by every
arm: same base snapshot, same arrival order, same injected drift edges at the
same steps, same batch roots, same neighbour-sampling seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor

from ..seeding import derive, numpy_rng
from .drift_process import DriftEvent, build_drift_schedule


@dataclass
class StreamPlan:
    """A fully materialised, replayable stream."""

    num_nodes: int
    base_edges: Tensor  # [2, E0] snapshot G_0
    stream_edges: Tensor  # [2, T] one arrival per step
    drift_events: List[DriftEvent]  # irregular, typed perturbations
    events_by_step: Dict[int, List[DriftEvent]]
    root_nodes: Tensor  # [T, B] batch roots per step
    sample_seeds: List[int]  # [T] neighbour-sampling seed per step
    seed: int

    @property
    def steps(self) -> int:
        return int(self.stream_edges.shape[1])

    @property
    def drift_steps(self) -> List[int]:
        return sorted(self.events_by_step)

    def edges_at(self, step: int) -> Tensor:
        return self.stream_edges[:, step : step + 1]

    def drift_at(self, step: int) -> List[DriftEvent]:
        return self.events_by_step.get(step, [])

    def describe(self) -> Dict[str, object]:
        kinds: Dict[str, int] = {}
        regimes: Dict[str, int] = {}
        for event in self.drift_events:
            kinds[event.kind] = kinds.get(event.kind, 0) + 1
            regimes[event.regime] = regimes.get(event.regime, 0) + 1

        gaps = np.diff(self.drift_steps) if len(self.drift_steps) > 1 else np.array([0])
        return {
            "seed": self.seed,
            "num_nodes": self.num_nodes,
            "base_undirected_pairs": int(self.base_edges.shape[1]),
            "steps": self.steps,
            "drift_events": len(self.drift_events),
            "kinds": kinds,
            "regimes": regimes,
            # Burstiness: for a fixed period this is 0, for Poisson ~1.
            "gap_mean": round(float(gaps.mean()), 1),
            "gap_cv": round(float(gaps.std() / gaps.mean()), 3) if gaps.mean() > 0 else 0.0,
            "batch_size": int(self.root_nodes.shape[1]),
        }


def build_plan(
    edge_index: Tensor,
    num_nodes: int,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    base_edge_fraction: float,
    drift_cfg,
    x: Tensor,
    y: Tensor,
    num_classes: int,
) -> StreamPlan:
    """Materialise a replayable stream from a dataset's edge index."""
    rng = numpy_rng(derive(seed, 0xB0))

    # Deduplicate to undirected pairs so the base/stream partition cannot put
    # (u,v) in the snapshot and (v,u) in the stream.
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    keep = lo != hi
    pairs = np.unique(np.stack([lo[keep], hi[keep]], axis=1), axis=0)

    order = rng.permutation(len(pairs))
    pairs = pairs[order]

    n_base = int(round(len(pairs) * base_edge_fraction))
    base = torch.from_numpy(pairs[:n_base].T.copy()).long()
    remaining = torch.from_numpy(pairs[n_base:].T.copy()).long()

    available = int(remaining.shape[1])
    if available == 0:
        raise ValueError("No edges left to stream; lower data.base_edge_fraction.")
    n_steps = min(steps, available)
    stream_edges = remaining[:, :n_steps]

    # Drift events: precomputed here so every arm replays identical drift.
    drift_events = build_drift_schedule(
        x=x,
        y=y,
        full_edge_index=edge_index,
        num_nodes=num_nodes,
        num_classes=num_classes,
        steps=n_steps,
        seed=seed,
        cfg=drift_cfg,
    )
    events_by_step: Dict[int, List[DriftEvent]] = {}
    for event in drift_events:
        events_by_step.setdefault(event.step, []).append(event)

    # Batch roots: fixed per step, shared by all arms. Sampled without
    # replacement within a step so each root appears once in the batch.
    root_rng = numpy_rng(derive(seed, 0x40))
    n_roots = min(batch_size, num_nodes)
    roots = np.stack(
        [root_rng.choice(num_nodes, size=n_roots, replace=False) for _ in range(n_steps)],
        axis=0,
    )
    root_nodes = torch.from_numpy(roots).long()

    sample_seeds = [derive(seed, 0x5A, step) for step in range(n_steps)]

    return StreamPlan(
        num_nodes=int(num_nodes),
        base_edges=base,
        stream_edges=stream_edges,
        drift_events=drift_events,
        events_by_step=events_by_step,
        root_nodes=root_nodes,
        sample_seeds=sample_seeds,
        seed=int(seed),
    )
