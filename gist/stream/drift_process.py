"""The drift process: when drift happens, where it lands, and what it changes.

Phase-1 measured that drift-triggered retraining was indistinguishable from
retraining on a timer. That was not a detector failure - it was forced by the
drift model. The prototype injected 20 random edges every 200 steps on a fixed
period, so a periodic retrainer was optimally matched to it *by construction*,
and no triggering policy can beat a timer when drift arrives on a timer.

Three changes make the comparison meaningful:

**Irregular, bursty arrivals.** A homogeneous Poisson process is not enough -
it still has a stable average rate, which a timer matches on average. Drift
arrives here via a regime-switching process alternating calm stretches (rare
events) with turbulent ones (frequent events). A fixed schedule must then
either over-retrain through the calm or under-react through the turbulence.

**Real concept drift.** Random edge insertions barely move a GAT. Events are
drawn from three kinds: structural (homophily-breaking edges), feature
(covariate shift toward another class's mean), and label (a cyclic class
remap, i.e. a genuine change in the target concept).

**Controllable locality.** ``localized`` events land on whole structural
communities; ``diffuse`` events hit an equal number of scattered nodes. The
diffuse setting is the honest control: GIST assumes drift is spatially
localized, so it should be shown to lose that advantage when the assumption
fails. See docs/deviations-from-proposal.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..drift.partitioners import assignment_to_groups, partition_graph
from ..seeding import derive, numpy_rng

KINDS = ("structural", "feature", "label")


@dataclass
class DriftEvent:
    """One scheduled perturbation of the stream."""

    step: int
    kind: str
    magnitude: float
    regime: str
    nodes: Tensor  # affected node ids
    num_classes: int
    edges: Optional[Tensor] = None  # structural: [2, k] edges to add
    feature_target: Optional[Tensor] = None  # feature: [F] donor mean vector
    label_shift: int = 0  # label: cyclic class offset

    def summary(self) -> dict:
        return {
            "step": self.step,
            "kind": self.kind,
            "regime": self.regime,
            "magnitude": round(float(self.magnitude), 3),
            "nodes": int(self.nodes.numel()),
            "edges": int(self.edges.shape[1]) if self.edges is not None else 0,
        }


# --- arrival processes -------------------------------------------------


def _regime_arrivals(rng: np.random.Generator, steps: int, cfg) -> List[tuple[int, str]]:
    """Alternating calm / turbulent regimes with exponential inter-arrivals."""
    arrivals: List[tuple[int, str]] = []
    t = 0.0
    regime = "calm"  # always open calm, so the detector sees a quiet baseline

    while t < steps:
        if regime == "calm":
            duration = float(rng.integers(cfg.calm_duration[0], cfg.calm_duration[1] + 1))
            mean_gap = cfg.calm_mean_gap
        else:
            duration = float(
                rng.integers(cfg.turbulent_duration[0], cfg.turbulent_duration[1] + 1)
            )
            mean_gap = cfg.turbulent_mean_gap

        end = min(t + duration, float(steps))
        cur = t + float(rng.exponential(mean_gap))
        while cur < end:
            arrivals.append((int(cur), regime))
            cur += float(rng.exponential(mean_gap))

        t = end
        regime = "turbulent" if regime == "calm" else "calm"

    return arrivals


def _poisson_arrivals(rng: np.random.Generator, steps: int, cfg) -> List[tuple[int, str]]:
    """Homogeneous Poisson - irregular, but with a stable rate."""
    arrivals: List[tuple[int, str]] = []
    cur = float(rng.exponential(cfg.poisson_mean_gap))
    while cur < steps:
        arrivals.append((int(cur), "poisson"))
        cur += float(rng.exponential(cfg.poisson_mean_gap))
    return arrivals


def _periodic_arrivals(steps: int, cfg) -> List[tuple[int, str]]:
    """The prototype's fixed-period schedule, kept for direct comparison."""
    return [(s, "periodic") for s in range(0, steps, cfg.interval)]


# --- spatial targeting -------------------------------------------------


def global_communities(
    edge_index: Tensor,
    num_nodes: int,
    seed: int,
    min_size: int = 20,
    backend: str = "auto",
    target_size: int = 2000,
) -> List[List[int]]:
    """Partition of the whole graph, used only to *target* drift.

    Deliberately not the partition the detector sees: the method must
    rediscover the drifted region from its own partition, so this does not
    hand it the answer.

    Goes through the shared partitioner rather than calling Louvain directly.
    This is built once per stream plan, but at OGBN-Arxiv scale a direct
    networkx Louvain here costs 240-390 s and silently dominated plan
    construction - the tracker had been moved to METIS while this path had
    not, so the plan was the bottleneck rather than the run.
    """
    assignment, _ = partition_graph(
        edge_index,
        num_nodes,
        backend=backend,
        seed=seed,
        target_size=target_size,
    )
    return assignment_to_groups(assignment, min_size) or [list(range(num_nodes))]


def _select_nodes(
    rng: np.random.Generator,
    locality: str,
    communities: Sequence[Sequence[int]],
    num_nodes: int,
    target_size: int,
) -> np.ndarray:
    target_size = max(1, min(target_size, num_nodes))

    if locality == "diffuse":
        return rng.choice(num_nodes, size=target_size, replace=False)

    picked: List[int] = []
    for idx in rng.permutation(len(communities)):
        picked.extend(communities[idx])
        if len(picked) >= target_size:
            break
    return np.asarray(picked[:target_size], dtype=np.int64)


def _select_nodes_weighted(
    rng: np.random.Generator,
    locality: str,
    communities: Sequence[Sequence[int]],
    community_mass: np.ndarray,
    weight: np.ndarray,
    target_size: int,
    pool_factor: float = 2.0,
) -> np.ndarray:
    """Preferential targeting: nodes (and communities) drawn with probability proportional to weight."""
    num_nodes = weight.size
    target_size = max(1, min(target_size, num_nodes))

    if locality == "diffuse":
        return rng.choice(num_nodes, size=target_size, replace=False, p=weight / weight.sum())

    pool: List[int] = []
    order = rng.choice(len(communities), size=len(communities), replace=False, p=community_mass / community_mass.sum())
    for idx in order:
        pool.extend(communities[idx])
        if len(pool) >= pool_factor * target_size:
            break
    pool_np = np.asarray(pool, dtype=np.int64)
    if pool_np.size <= target_size:
        return pool_np
    w = weight[pool_np]
    return pool_np[rng.choice(pool_np.size, size=target_size, replace=False, p=w / w.sum())]


def _preferential_edges(rng, nodes_np, weight, labels, count) -> Tensor:
    """Homophily-breaking edges whose endpoints are both drawn by weight (preferential attachment)."""
    w = weight[nodes_np]
    src = rng.choice(nodes_np, size=count, replace=True, p=w / w.sum())
    cand = rng.choice(weight.size, size=(count, 8), replace=True, p=weight / weight.sum())
    other = labels[cand] != labels[src][:, None]
    first = np.where(other.any(axis=1), other.argmax(axis=1), 0)
    dst = cand[np.arange(count), first]
    return torch.from_numpy(np.stack([src.astype(np.int64), dst.astype(np.int64)], axis=0))


# --- schedule construction ---------------------------------------------


def build_drift_schedule(
    *,
    x: Tensor,
    y: Tensor,
    full_edge_index: Tensor,
    num_nodes: int,
    num_classes: int,
    steps: int,
    seed: int,
    cfg,
) -> List[DriftEvent]:
    rng = numpy_rng(derive(seed, 0xDF))

    if cfg.mode == "regime":
        arrivals = _regime_arrivals(rng, steps, cfg)
    elif cfg.mode == "poisson":
        arrivals = _poisson_arrivals(rng, steps, cfg)
    elif cfg.mode == "periodic":
        arrivals = _periodic_arrivals(steps, cfg)
    else:
        raise ValueError(f"Unknown drift mode {cfg.mode!r}")

    communities: List[List[int]] = []
    if cfg.locality == "localized":
        communities = global_communities(
            full_edge_index,
            num_nodes,
            seed=seed,
            backend=getattr(cfg, "partitioner", "auto"),
            target_size=getattr(cfg, "partition_target_size", 2000),
        )

    labels = y.numpy()
    events: List[DriftEvent] = []

    bias = float(getattr(cfg, "degree_bias", 0.0))
    if bias > 0:
        degree = np.bincount(full_edge_index[0].numpy(), minlength=num_nodes).astype(np.float64)
        weight = (degree + 1.0) ** bias
        community_mass = np.array([weight[c].sum() for c in communities]) if communities else None
        pa_rng = numpy_rng(derive(seed, 0xDA))

    for step, regime in arrivals:
        kind = str(rng.choice(list(cfg.kinds)))
        magnitude = float(rng.uniform(cfg.magnitude[0], cfg.magnitude[1]))
        fraction = float(rng.uniform(cfg.affected_fraction[0], cfg.affected_fraction[1]))
        nodes_np = _select_nodes(
            rng, cfg.locality, communities, num_nodes, int(round(fraction * num_nodes))
        )
        nodes = torch.from_numpy(np.asarray(nodes_np, dtype=np.int64))

        event = DriftEvent(
            step=step,
            kind=kind,
            magnitude=magnitude,
            regime=regime,
            nodes=nodes,
            num_classes=num_classes,
        )

        if kind == "structural":
            # Homophily-breaking edges: connect affected nodes to nodes of a
            # *different* class. Uniformly random edges barely perturb a GAT;
            # these attack the assumption message passing relies on.
            count = int(round(cfg.structural_edges * (0.5 + magnitude)))
            src = rng.choice(nodes_np, size=count, replace=True)
            dst = np.empty(count, dtype=np.int64)
            for i, s in enumerate(src):
                for _ in range(8):
                    cand = int(rng.integers(0, num_nodes))
                    if labels[cand] != labels[s]:
                        dst[i] = cand
                        break
                else:
                    dst[i] = int(rng.integers(0, num_nodes))
            event.edges = torch.from_numpy(np.stack([src.astype(np.int64), dst], axis=0))

        elif kind == "feature":
            # Covariate shift: interpolate toward another class's mean feature.
            own = int(np.bincount(labels[nodes_np], minlength=num_classes).argmax())
            choices = [c for c in range(num_classes) if c != own] or [own]
            donor_class = int(rng.choice(choices))
            donor_mask = torch.from_numpy(labels == donor_class)
            donor = (
                x[donor_mask].mean(dim=0) if bool(donor_mask.any()) else x.mean(dim=0)
            )
            event.feature_target = donor.clone()

        elif kind == "label":
            # Genuine concept change: the target function itself moves.
            event.label_shift = int(rng.integers(1, max(2, num_classes)))

        else:
            raise ValueError(f"Unknown drift kind {kind!r}")

        if bias > 0:
            # The uniform draws above still run, so the main rng (and with it every event's time, kind,
            # magnitude and size) is the same as at alpha = 0; only which nodes and edges are hit changes.
            nodes_np = _select_nodes_weighted(pa_rng, cfg.locality, communities, community_mass, weight,
                                              int(event.nodes.numel()))
            event.nodes = torch.from_numpy(np.asarray(nodes_np, dtype=np.int64))
            if kind == "structural":
                event.edges = _preferential_edges(pa_rng, nodes_np, weight, labels, int(event.edges.shape[1]))
            elif kind == "feature":
                own = int(np.bincount(labels[nodes_np], minlength=num_classes).argmax())
                choices = [c for c in range(num_classes) if c != own] or [own]
                donor_mask = torch.from_numpy(labels == int(pa_rng.choice(choices)))
                event.feature_target = (x[donor_mask].mean(dim=0) if bool(donor_mask.any()) else x.mean(dim=0)).clone()

        events.append(event)

    return events


# --- application --------------------------------------------------------


def apply_drift(event: DriftEvent, x: Tensor, y: Tensor, graph_state) -> None:
    """Apply one event in place. Depends only on the plan, never on the model.

    That independence is what keeps arms paired: every arm sees an identical
    sequence of graph, feature and label states.
    """
    if event.kind == "structural":
        if event.edges is not None:
            graph_state.add_edges(event.edges)

    elif event.kind == "feature":
        if event.feature_target is not None:
            lam = float(event.magnitude)
            idx = event.nodes
            x[idx] = (1.0 - lam) * x[idx] + lam * event.feature_target.unsqueeze(0)

    elif event.kind == "label":
        idx = event.nodes
        y[idx] = (y[idx] + event.label_shift) % event.num_classes
