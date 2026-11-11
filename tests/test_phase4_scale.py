"""Phase 4 tests: partitioner backends and large-graph readiness.

networkx Louvain over OGBN-Arxiv extrapolates to 240-390 s per refresh, which
makes the experiment infeasible rather than merely slow. These tests pin the
behaviour of the METIS path that replaces it, and guard the properties the
community layer relies on regardless of which backend produced the partition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.utils import to_undirected

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gist.drift.global_communities import GlobalCommunityTracker
from gist.drift.partitioners import (
    AUTO_METIS_EDGE_THRESHOLD,
    assignment_to_groups,
    metis_available,
    partition_graph,
    resolve_num_parts,
)

requires_metis = pytest.mark.skipif(not metis_available(), reason="METIS unavailable")


def _graph(num_nodes: int, num_edges: int, seed: int = 0) -> torch.Tensor:
    """Erdos-Renyi: no community structure."""
    rng = np.random.default_rng(seed)
    ei = torch.from_numpy(
        np.stack([rng.integers(0, num_nodes, num_edges),
                  rng.integers(0, num_nodes, num_edges)], axis=0)
    ).long()
    return to_undirected(ei, num_nodes=num_nodes)


def _clustered_graph(
    num_nodes: int, num_blocks: int, num_edges: int, p_in: float = 0.9, seed: int = 0
) -> torch.Tensor:
    """Stochastic block model - a graph that actually has communities.

    Partition stability is only a meaningful property on a graph with
    structure to be stable about; on Erdos-Renyi any balanced partition is
    arbitrary and reshuffles freely.
    """
    rng = np.random.default_rng(seed)
    block = rng.integers(0, num_blocks, num_nodes)
    src = rng.integers(0, num_nodes, num_edges)
    same = rng.random(num_edges) < p_in
    dst = np.empty(num_edges, dtype=np.int64)
    for b in range(num_blocks):
        members = np.flatnonzero(block == b)
        pick = same & (block[src] == b)
        if pick.any() and members.size:
            dst[pick] = rng.choice(members, size=int(pick.sum()))
    rest = ~same
    dst[rest] = rng.integers(0, num_nodes, int(rest.sum()))
    ei = torch.from_numpy(np.stack([src, dst], axis=0)).long()
    return to_undirected(ei, num_nodes=num_nodes)


# --- backend selection --------------------------------------------------


def test_auto_prefers_louvain_on_small_graphs():
    ei = _graph(500, 2000)
    assert ei.shape[1] < AUTO_METIS_EDGE_THRESHOLD
    _, backend = partition_graph(ei, 500, backend="auto")
    assert backend == "louvain", "small graphs should keep the Louvain path"


@requires_metis
def test_auto_switches_to_metis_on_large_graphs():
    nodes = 60_000
    ei = _graph(nodes, AUTO_METIS_EDGE_THRESHOLD)
    assert ei.shape[1] > AUTO_METIS_EDGE_THRESHOLD
    _, backend = partition_graph(ei, nodes, backend="auto", target_size=2000)
    assert backend == "metis"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        partition_graph(_graph(100, 200), 100, backend="spectral")


# --- partition properties ----------------------------------------------


@requires_metis
def test_metis_assigns_every_node_to_a_part():
    nodes = 5_000
    assignment, _ = partition_graph(
        _graph(nodes, 20_000), nodes, backend="metis", num_parts=16
    )
    assert assignment.shape == (nodes,)
    assert int(assignment.min()) >= 0, "METIS should leave no node unassigned"
    assert int(assignment.max()) < 16


@requires_metis
def test_metis_parts_are_balanced():
    """Balanced parts are why retraining-set size is predictable under METIS."""
    nodes = 20_000
    assignment, _ = partition_graph(
        _graph(nodes, 80_000), nodes, backend="metis", num_parts=20
    )
    sizes = torch.bincount(assignment).float()
    assert float(sizes.max() / sizes.min()) < 2.0, (
        f"parts ranged {int(sizes.min())}..{int(sizes.max())}"
    )


def test_resolve_num_parts_derives_from_target_size():
    assert resolve_num_parts(169_343, None, 2000) == 85
    assert resolve_num_parts(169_343, 32, 2000) == 32, "explicit k must win"
    assert resolve_num_parts(100, None, 10_000) >= 2, "never fewer than 2 parts"


def test_assignment_to_groups_drops_small_communities():
    assignment = torch.tensor([0, 0, 0, 1, 2, 2, 2, 2, -1])
    groups = assignment_to_groups(assignment, min_size=3)
    assert sorted(len(g) for g in groups) == [3, 4]
    assert all(-1 not in g for g in groups)


# --- tracker integration ------------------------------------------------


@requires_metis
def test_tracker_uses_metis_and_keeps_ids_stable():
    """Ids must survive a refresh, or per-community drift is not a time series.

    Measured on a clustered graph: METIS on a structureless graph reshuffles
    freely and stability is meaningless there.
    """
    nodes, blocks = 20_000, 20
    ei = _clustered_graph(nodes, blocks, 200_000)
    tracker = GlobalCommunityTracker(
        nodes, partitioner="metis", num_parts=blocks, min_size=10, seed=0
    )

    first = tracker.refresh(ei, step=0)
    assert tracker.backend_used == "metis"
    assert first.num_communities > 0

    # The stream adds a fraction of a percent of new edges over a run.
    extra = _clustered_graph(nodes, blocks, 2_000, seed=1)
    second = tracker.refresh(torch.cat([ei, extra], dim=1), step=200)

    assert second.stability > 0.5, (
        f"only {second.stability:.1%} of nodes kept their community id"
    )


@requires_metis
def test_community_nodes_respects_restrict_mask_and_limit():
    nodes = 10_000
    tracker = GlobalCommunityTracker(nodes, partitioner="metis", num_parts=10, seed=0)
    tracker.refresh(_graph(nodes, 40_000), step=0)

    trainable = torch.zeros(nodes, dtype=torch.bool)
    trainable[: nodes // 2] = True
    cids = list(tracker.partition.sizes)[:2]

    picked = tracker.community_nodes(cids, restrict_mask=trainable, limit=64)
    assert picked.numel() <= 64
    assert bool(trainable[picked].all()), "returned an untrainable node"
    assert bool(torch.isin(tracker.partition.assignment[picked],
                           torch.tensor(cids)).all())
